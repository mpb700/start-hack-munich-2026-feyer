"""
Fast host-side simulation path: sparse LU solves in SciPy and a hand-written adjoint.

`differometor.simulate.simulate` dispatches here for setups without parameter linking,
without homodyne detectors (qhd), with exactly one quantum noise detector and a single
carrier operating point. Everything else uses the original JAX implementation.

How it works
------------
- Matrix entries are produced by the unchanged component functions of
  `differometor.components`, re-bound to a tiny NumPy shim that evaluates them on dual
  numbers (value + derivative with respect to every real input). No XLA compilation and
  no hand-written derivatives of the physics formulas are needed.
- The carrier system and the signal system of every frequency are assembled as sparse
  CSC matrices and factorized with SuperLU, loaded directly from scipy's compiled
  `_superlu` extension (`differometor.host_superlu`) so that `scipy` is never imported.
- The quantum noise of the (single) detector is `Re(y^H N y)` with `y = S^-H w`; the
  same solve is reused as the adjoint helper for the gradient.
- Reverse mode is written by hand. Cotangents follow JAX's convention for complex
  values (bilinear pairing), so signal adjoints solve with `S^-T`, not `S^-H`.

The JAX integration (`simulate_fast`) is a `jax.custom_vjp` around two
`jax.pure_callback`s. The callbacks keep no Python state: the forward callback returns
every residual the backward pass needs as arrays that flow through JAX, and the backward
callback rebuilds the sparsity plan and refactorizes the matrices from them.
"""

import jax
import jax.numpy as jnp
import numpy as np
from differometor.host_superlu import CSC as _HostCSC, splu as _host_splu

import differometor.components as components
from differometor.components import UNIT_VACUUM, H_PLANCK, F0

NOISE_SCALE = UNIT_VACUUM * H_PLANCK * F0 * 0.25
# Columns of the detector basis Z = S^-T [e_d] kept per frequency (upper + lower sideband of one port).
DETECTOR_WIDTH = 2


# ---------------------------------------------------------------------
# Component functions on dual numbers
# ---------------------------------------------------------------------


class Dual:
    """Value plus derivatives with respect to k real inputs; `tan` has shape (k, *val.shape) or is 0."""

    __slots__ = ("val", "tan")
    __array_priority__ = 1000

    def __init__(self, val, tan=0.0):
        self.val, self.tan = val, tan

    @staticmethod
    def lift(x):
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    def __neg__(self):
        return Dual(-self.val, -self.tan)

    def __add__(self, other):
        other = Dual.lift(other)
        return Dual(self.val + other.val, self.tan + other.tan)

    __radd__ = __add__

    def __sub__(self, other):
        other = Dual.lift(other)
        return Dual(self.val - other.val, self.tan - other.tan)

    def __rsub__(self, other):
        other = Dual.lift(other)
        return Dual(other.val - self.val, other.tan - self.tan)

    def __mul__(self, other):
        other = Dual.lift(other)
        return Dual(self.val * other.val, self.tan * other.val + self.val * other.tan)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual.lift(other)
        quotient = self.val / other.val
        return Dual(quotient, (self.tan - quotient * other.tan) / other.val)

    def __rtruediv__(self, other):
        return Dual.lift(other).__truediv__(self)

    def __pow__(self, exponent):
        if isinstance(exponent, Dual):
            raise TypeError("Dual exponents are not supported")
        return Dual(self.val**exponent, exponent * self.val ** (exponent - 1) * self.tan)


def _unary(function, derivative):
    def wrapped(x):
        if not isinstance(x, Dual):
            return function(x)
        y = function(x.val)
        return Dual(y, derivative(x.val, y) * x.tan)

    return wrapped


class _NumpyShim:
    """Stand-in for `jax.numpy` inside the component functions (all inputs are real)."""

    pi, e, complex128, ndarray = np.pi, np.e, np.complex128, np.ndarray
    log10, log = staticmethod(np.log10), staticmethod(np.log)
    exp = staticmethod(_unary(np.exp, lambda x, y: y))
    sqrt = staticmethod(_unary(np.sqrt, lambda x, y: 0.5 / y))
    sin = staticmethod(_unary(np.sin, lambda x, y: np.cos(x)))
    cos = staticmethod(_unary(np.cos, lambda x, y: -np.sin(x)))
    sinh = staticmethod(_unary(np.sinh, lambda x, y: np.cosh(x)))
    cosh = staticmethod(_unary(np.cosh, lambda x, y: np.sinh(x)))
    arcsin = staticmethod(_unary(np.arcsin, lambda x, y: 1.0 / np.sqrt(1.0 - x * x)))
    radians = staticmethod(_unary(np.radians, lambda x, y: np.pi / 180.0))

    @staticmethod
    def conj(x):  # valid because every primal input is real
        return Dual(np.conj(x.val), np.conj(x.tan)) if isinstance(x, Dual) else np.conj(x)

    conjugate = conj

    @staticmethod
    def abs(x):
        if not isinstance(x, Dual):
            return np.abs(x)
        magnitude = np.abs(x.val)
        return Dual(magnitude, (np.conj(x.val) * x.tan).real / magnitude)

    @staticmethod
    def array(items, dtype=None):
        return list(items)


class _Output(list):
    """Component output vector; supports the `-standardize_output(...)` sign flip."""

    def __neg__(self):
        return _Output(-x for x in self)


def _rebind_component_functions():
    """Copies of `components.FUNCTIONS` whose global `jnp` is the dual-number NumPy shim."""
    standardize = lambda output: _Output(list(output) + [0.0] * (4 - len(output)))
    namespace = components.rebind_functions(_NumpyShim, {"standardize_output": standardize})
    return [namespace[function.__name__] for function in components.FUNCTIONS]


FUNCTIONS_NUMPY = _rebind_component_functions()


def _evaluate(function_index, inputs):
    """inputs: (..., rows, k) -> values (..., rows, 4) and Jacobians (..., rows, 4, k)."""
    lead, width = inputs.shape[:-1], inputs.shape[-1]
    seed = np.eye(width)
    duals = [Dual(inputs[..., j], seed[j].reshape((width,) + (1,) * len(lead))) for j in range(width)]
    outputs = FUNCTIONS_NUMPY[function_index](duals)
    values = np.empty(lead + (4,), complex)
    jacobian = np.zeros(lead + (4, width), complex)
    for o, item in enumerate(outputs):
        if isinstance(item, Dual):
            values[..., o] = item.val
            if not np.isscalar(item.tan):
                jacobian[..., o, :] = np.moveaxis(np.broadcast_to(item.tan, (width,) + lead), 0, -1)
        else:
            values[..., o] = item
    return values, jacobian


def _evaluate_values(function_index, inputs):
    """Values only (no derivatives): inputs (..., rows, k) -> (..., rows, 4)."""
    outputs = FUNCTIONS_NUMPY[function_index]([inputs[..., j] for j in range(inputs.shape[-1])])
    values = np.empty(inputs.shape[:-1] + (4,), complex)
    for o, item in enumerate(outputs):
        values[..., o] = item
    return values


class _EntryBlock:
    """One group of matrix entries (carrier, signal or noise) produced by component functions.

    `changing` lists the parameter columns that differ between the rows (the frequency sweep).
    Calls that read none of them give the same entries in every row and are evaluated once.
    """

    def __init__(self, function_input_indices, function_indices, output_indices, changing=()):
        self.inputs = np.asarray(function_input_indices)
        function_indices = np.asarray(function_indices)
        self.outputs = np.asarray(output_indices)
        calls = self.inputs.shape[0]
        self.varying = np.isin(self.inputs, np.asarray(changing, int)).any(axis=1)
        self.groups = []  # (function index, members, varying) with members sorted by call
        for k in np.unique(function_indices):
            for varying in (True, False):
                members = np.nonzero((function_indices == k) & (self.varying == varying))[0]
                if members.size:
                    self.groups.append((int(k), members, varying))
        # Scatter from entries to (call, output) slots; several entries may share a slot.
        self.flat = self.outputs[0] * 4 + self.outputs[1]

    def _gather(self, values):
        return values[:, self.outputs[0], self.outputs[1]]

    def values(self, parameters):
        """parameters: (R, P) -> entries (R, E), no derivatives."""
        rows, calls = parameters.shape[0], self.inputs.shape[0]
        values = np.empty((rows, calls, 4), complex)
        for function_index, members, varying in self.groups:
            source = parameters if varying else parameters[:1]
            values[:, members] = _evaluate_values(function_index, source[:, self.inputs[members]])
        return self._gather(values)

    def run(self, parameters):
        """parameters: (R, P). Returns entries (R, E) and the Jacobian tape for the reverse pass."""
        rows, calls, width = parameters.shape[0], self.inputs.shape[0], self.inputs.shape[1]
        values = np.empty((rows, calls, 4), complex)
        tape = []
        for function_index, members, varying in self.groups:
            source = parameters if varying else parameters[:1]
            values[:, members], jacobian = _evaluate(function_index, source[:, self.inputs[members]])
            tape.append(jacobian)
        return self._gather(values), tape

    def back(self, tape, entries_bar, parameter_count):
        """entries_bar: (R, E) -> real parameter cotangent summed over the rows (P,)."""
        rows, calls = entries_bar.shape[0], self.inputs.shape[0]
        outputs_bar = np.zeros((rows, calls * 4), dtype=entries_bar.dtype)
        np.add.at(outputs_bar, (np.arange(rows)[:, None], self.flat[None, :]), entries_bar)
        outputs_bar = outputs_bar.reshape(rows, calls, 4)
        parameters_bar = np.zeros(parameter_count)
        for (function_index, members, varying), jacobian in zip(self.groups, tape):
            if varying:
                inputs_bar = np.einsum("rco,rcoj->cj", outputs_bar[:, members], jacobian).real
            else:  # identical Jacobian in every row
                inputs_bar = np.einsum("co,coj->cj", outputs_bar[:, members].sum(0), jacobian[0]).real
            parameters_bar += np.bincount(self.inputs[members].ravel(), inputs_bar.ravel(), parameter_count)
        return parameters_bar


class EntryPlan:
    """Parameter expansion and component evaluation (mirrors the start of `simulate`)."""

    def __init__(self, A):
        parameters = np.asarray(A["parameters"])
        if np.any(parameters.imag != 0):
            raise ValueError("fast path requires real base parameters")
        self.base = parameters[0].real.copy()
        self.optimized_parameter_indices = np.asarray(A["optimized_parameter_indices"])
        self.optimized_value_indices = np.asarray(A["optimized_value_indices"])
        self.carrier_changing_indices = np.asarray(A["carrier_changing_parameter_indices"])
        self.carrier_changing_values = np.asarray(A["carrier_changing_values"])
        self.signal_changing_indices = np.asarray(A["signal_changing_parameter_indices"])
        self.signal_changing_values = np.asarray(A["signal_changing_values"])
        for values in (self.carrier_changing_values, self.signal_changing_values):
            if np.any(np.imag(values) != 0):
                raise ValueError("fast path requires real changing values")
        changing = self.signal_changing_indices
        self.carrier = _EntryBlock(A["carrier_function_input_indices"], A["carrier_function_indices"], A["carrier_output_indices"])
        self.signal = _EntryBlock(A["signal_function_input_indices"], A["signal_function_indices"], A["signal_output_indices"], changing)
        self.noise = _EntryBlock(A["noise_function_input_indices"], A["noise_function_indices"], A["noise_output_indices"], changing)

    def _rows(self, optimized_parameters):
        P = self.base.copy()
        P[self.optimized_parameter_indices] = np.asarray(optimized_parameters, float)[self.optimized_value_indices]
        carrier_rows = np.tile(P, (self.carrier_changing_values.shape[1], 1))
        carrier_rows[:, self.carrier_changing_indices] = np.real(self.carrier_changing_values).T
        signal_rows = np.tile(carrier_rows[0], (self.signal_changing_values.shape[1], 1))
        signal_rows[:, self.signal_changing_indices] = np.real(self.signal_changing_values).T
        return carrier_rows[:1], signal_rows

    def values(self, optimized_parameters):
        """Carrier, signal and noise entries without derivatives (forward pass)."""
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        return self.carrier.values(carrier_rows)[0], self.signal.values(signal_rows), self.noise.values(signal_rows)

    def entries(self, optimized_parameters):
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        carrier, carrier_tape = self.carrier.run(carrier_rows)
        signal, signal_tape = self.signal.run(signal_rows)
        noise, noise_tape = self.noise.run(signal_rows)
        return (carrier[0], signal, noise), (carrier_tape, signal_tape, noise_tape)

    def pullback(self, tape, carrier_bar, signal_bar, noise_bar, value_count):
        carrier_tape, signal_tape, noise_tape = tape
        parameter_count = self.base.size
        signal_rows_bar = self.signal.back(signal_tape, signal_bar, parameter_count) + self.noise.back(
            noise_tape, noise_bar, parameter_count
        )
        signal_rows_bar[self.signal_changing_indices] = 0.0  # overwritten by the sweep values
        P_bar = self.carrier.back(carrier_tape, carrier_bar[None], parameter_count) + signal_rows_bar
        P_bar[self.carrier_changing_indices] = 0.0
        return np.bincount(self.optimized_value_indices, P_bar[self.optimized_parameter_indices], value_count)


# ---------------------------------------------------------------------
# Sparse linear systems
# ---------------------------------------------------------------------


def _last_writer(rows, columns, width):
    """Mask of scatter entries that survive `.at[rows, columns].set(...)` (last write wins)."""
    flat = rows * width + columns
    reverse_first = np.unique(flat[::-1], return_index=True)[1]
    mask = np.zeros(flat.size, bool)
    mask[flat.size - 1 - reverse_first] = True
    return mask


class SystemPlan:
    """Static sparsity data derived from the index arrays (cheap, rebuilt in every callback)."""

    def __init__(self, A):
        get = lambda key: np.asarray(A[key])
        self.carrier_base = get("carrier_matrix")
        self.n = n = self.carrier_base.shape[0]
        signal_base = get("signal_matrix")
        self.m = m = signal_base.shape[0]

        # Carrier system: pattern of the n x n matrix, right-hand side in column n.
        self.cmi = get("carrier_matrix_indices") % np.array([[n], [n + 1]])
        self.carrier_live = _last_writer(self.cmi[0], self.cmi[1], n + 1)
        mask = self.carrier_base[:, :-1] != 0
        in_matrix = self.cmi[1] < n
        mask[self.cmi[0][in_matrix], self.cmi[1][in_matrix]] = True
        self.carrier_cols, self.carrier_rows = np.nonzero(mask.T)
        self.carrier_ptr = np.concatenate([[0], np.cumsum(mask.sum(0))])

        # Signal system: pattern of the augmented m x (m + 1) matrix in CSC order.
        smi = get("signal_matrix_indices") % np.array([[m], [m + 1]])
        placing = get("signal_carrier_placing_indices") % np.array([[m], [m + 1]])
        self.signal_carrier_indices = get("signal_carrier_indices").ravel()
        mask = signal_base != 0
        mask[smi[0], smi[1]] = True
        mask[placing[0], placing[1]] = True
        cols, rows = np.nonzero(mask.T)
        self.rows, self.cols, self.size = rows, cols, rows.size
        position = -np.ones(mask.shape, int)
        position[rows, cols] = np.arange(self.size)
        self.signal_base = signal_base[rows, cols]
        self.update_positions = position[smi[0], smi[1]]
        self.update_live = _last_writer(smi[0], smi[1], m + 1)
        self.placing_positions = position[placing[0], placing[1]]
        self.placing_live = _last_writer(placing[0], placing[1], m + 1)
        # Entries that are complex-conjugated for the lower sideband (see `simulate`).
        self.conjugated = (
            ((rows >= n) & (rows < 2 * n) & (cols >= n) & (cols < 2 * n))
            | ((rows >= 2 * n) & (cols < n))
            | ((rows >= n) & (rows < 2 * n) & (cols >= 2 * n))
        )
        self.in_matrix = cols < m
        self.matrix_rows = rows[self.in_matrix].astype(np.int32)
        self.matrix_cols = cols[self.in_matrix]
        self.matrix_ptr = np.concatenate([[0], np.cumsum(mask[:, :m].sum(0))])
        self.rhs_rows = rows[~self.in_matrix]

        # Noise matrix: base entries plus entries written by noise component functions.
        noise_base = get("noise_matrix")
        nmi = get("noise_system_matrix_indices") % m
        self.noise_live = _last_writer(nmi[0], nmi[1], m)
        written = np.zeros((m, m), bool)
        written[nmi[0], nmi[1]] = True
        base_rows, base_cols = np.nonzero((noise_base != 0) & ~written)
        self.noise_rows = np.concatenate([base_rows, nmi[0][self.noise_live]])
        self.noise_cols = np.concatenate([base_cols, nmi[1][self.noise_live]])
        self.noise_base = noise_base[base_rows, base_cols]
        self.selection = get("noise_selection_vectors")[0, 0]
        # Rows the quantum noise detector reads. When they fit into DETECTOR_WIDTH, the forward pass
        # solves S^T z = e_d for each of them; Y and the signal adjoint are then combinations of z.
        self.detector = np.nonzero(self.selection)[0]
        self.use_basis = 0 < self.detector.size <= DETECTOR_WIDTH
        self.order = None

    def carrier_matrix(self, carrier_entries):
        n = self.n
        M = self.carrier_base.astype(complex)
        M[self.cmi[0][self.carrier_live], self.cmi[1][self.carrier_live]] = carrier_entries[self.carrier_live]
        A = _HostCSC(M[self.carrier_rows, self.carrier_cols], self.carrier_rows, self.carrier_ptr, (n, n))
        return A, M[:, -1]

    def signal_values(self, signal_entries, carrier):
        """Nonzero values of the augmented signal matrices, shape (F, size), plus placing pre-values."""
        F = signal_entries.shape[0]
        V = np.broadcast_to(self.signal_base.astype(complex), (F, self.size)).copy()
        V[:, self.update_positions[self.update_live]] = signal_entries[:, self.update_live]
        pre = V[:, self.placing_positions].copy()
        live = self.placing_live
        V[:, self.placing_positions[live]] = -pre[:, live] * carrier[self.signal_carrier_indices[live]]
        V[:, self.conjugated] = np.conj(V[:, self.conjugated])
        return V, pre

    def _set_order(self, values_row):
        """Column order from COLAMD on one matrix, reused for all frequencies (shared pattern)."""
        first = _HostCSC(values_row[self.in_matrix], self.matrix_rows, self.matrix_ptr, (self.m, self.m))
        order = np.argsort(_host_splu(first).perm_c)
        counts = np.diff(self.matrix_ptr)[order]
        self.ordered_ptr = np.concatenate([[0], np.cumsum(counts)])
        gather = np.repeat(self.matrix_ptr[order] - self.ordered_ptr[:-1], counts) + np.arange(self.ordered_ptr[-1])
        self.ordered_rows = self.matrix_rows[gather]
        self.ordered_values = np.nonzero(self.in_matrix)[0][gather]
        self.order = order

    def signal_lu(self, values_row):
        if self.order is None:
            self._set_order(values_row)
        B = _HostCSC(values_row[self.ordered_values], self.ordered_rows, self.ordered_ptr, (self.m, self.m))
        return _OrderedLU(_host_splu(B, permc_spec="NATURAL"), self.order)

    def noise_values(self, noise_entries):
        F = noise_entries.shape[0]
        base = np.broadcast_to(self.noise_base.astype(complex), (F, self.noise_base.size))
        return np.concatenate([base, noise_entries[:, self.noise_live]], axis=1)

    def detector_weights(self, carrier):
        n, s = self.n, self.selection
        w = s.astype(complex).copy()
        w[:n] = s[:n] * carrier
        w[n : 2 * n] = np.conj(s[n : 2 * n] * carrier)
        return w


class _OrderedLU:
    """LU of B = A[:, order]; solves with A, A^T and A^H."""

    def __init__(self, lu, order):
        self.lu, self.order = lu, order

    def solve(self, b, trans="N"):
        if trans == "N":
            y = self.lu.solve(b)
            x = np.empty_like(y)
            x[self.order] = y
            return x
        return self.lu.solve(b[self.order], trans=trans)


def _matvec(rows, cols, values, vector, size):
    """y = A x for a sparse matrix given as (rows, cols, values)."""
    product = values * vector[cols]
    return np.bincount(rows, product.real, size) + 1j * np.bincount(rows, product.imag, size)


def forward(plan, carrier_entries, signal_entries, noise_entries):
    """Carrier, signal and noise outputs plus the adjoint helpers Y = S^-H w, MU and Z.

    Z[f] = S_f^-T [e_d for d in plan.detector] (zero-padded to DETECTOR_WIDTH columns). It gives
    Y = conj(Z conj(w_d)) here and the signal adjoint S^-T signal_bar in `backward` without a
    second factorization, whenever the cotangent of the signal lives on the detector rows.
    """
    m = plan.m
    carrier_matrix, carrier_rhs = plan.carrier_matrix(carrier_entries)
    carrier = _host_splu(carrier_matrix).solve(carrier_rhs)
    V, _ = plan.signal_values(signal_entries, carrier)
    w = plan.detector_weights(carrier)
    N = plan.noise_values(noise_entries)
    F = V.shape[0]
    rhs = np.zeros((F, m), complex)
    rhs[:, plan.rhs_rows] = V[:, ~plan.in_matrix]
    X = np.empty((F, m), complex)
    Y = np.empty((F, m), complex)
    MU = np.empty((F, m), complex)
    Z = np.zeros((F, m, DETECTOR_WIDTH), complex)
    power = np.empty(F)
    d, k = plan.detector, plan.detector.size
    basis = np.zeros((m, k), complex)
    basis[d, np.arange(k)] = 1.0
    w_d = np.conj(w[d])
    for f in range(F):
        lu = plan.signal_lu(V[f])
        X[f] = lu.solve(rhs[f])
        if plan.use_basis:
            z = lu.solve(basis, trans="T")
            Z[f, :, :k] = z
            y = np.conj(z @ w_d)
        else:
            y = lu.solve(w, trans="H")
        q = _matvec(plan.noise_rows, plan.noise_cols, N[f], y, m)  # N y
        qh = _matvec(plan.noise_cols, plan.noise_rows, np.conj(N[f]), y, m)  # N^H y
        power[f] = (np.conj(y) @ q).real
        MU[f] = lu.solve(0.5 * (q + qh))
        Y[f] = y
    noise = np.sqrt(2 * power * NOISE_SCALE)
    return carrier, X, noise, Y, MU, Z


def _signal_adjoint(plan, V, Z, signal_bar):
    """lam_f = S_f^-T signal_bar_f for every frequency, shape (F, m)."""
    support = np.nonzero(np.any(signal_bar != 0, axis=0))[0]
    d, k = plan.detector, plan.detector.size
    if plan.use_basis and np.isin(support, d).all():
        return np.einsum("fmk,fk->fm", Z[:, :, :k], signal_bar[:, d])
    lam = np.empty(signal_bar.shape, complex)
    for f in range(V.shape[0]):
        lam[f] = plan.signal_lu(V[f]).solve(signal_bar[f].astype(complex), trans="T")
    return lam


def backward(plan, carrier_entries, signal_entries, carrier, X, noise, Y, MU, Z, carrier_bar, signal_bar, noise_bar):
    """Cotangents of the carrier, signal and noise entries (JAX's bilinear convention)."""
    n, m = plan.n, plan.m
    carrier_matrix, _ = plan.carrier_matrix(carrier_entries)
    V, pre = plan.signal_values(signal_entries, carrier)
    F = V.shape[0]
    carrier_bar = carrier_bar.astype(complex).copy()
    V_bar = np.empty((F, plan.size), complex)
    g = noise_bar * NOISE_SCALE / noise  # d noise / d power
    rows, cols = plan.rows[plan.in_matrix], plan.cols[plan.in_matrix]
    lam = _signal_adjoint(plan, V, Z, signal_bar)
    V_bar[:, plan.in_matrix] = -lam[:, rows] * X[:, cols] - 2 * g[:, None] * np.conj(Y[:, rows]) * MU[:, cols]
    V_bar[:, ~plan.in_matrix] = lam[:, plan.rhs_rows]
    w_bar = 2 * (g[:, None] * np.conj(MU)).sum(0)

    N_bar = g[:, None] * np.conj(Y[:, plan.noise_rows]) * Y[:, plan.noise_cols]
    noise_entries_bar = np.zeros((F, plan.noise_live.size), complex)
    noise_entries_bar[:, plan.noise_live] = N_bar[:, plan.noise_base.size :]

    V_bar[:, plan.conjugated] = np.conj(V_bar[:, plan.conjugated])
    live = plan.placing_live
    placed = plan.placing_positions[live]
    sci = plan.signal_carrier_indices[live]
    np.add.at(carrier_bar, sci, -(V_bar[:, placed] * pre[:, live]).sum(0))
    V_bar[:, placed] = -V_bar[:, placed] * carrier[sci]
    signal_entries_bar = np.zeros_like(signal_entries)
    signal_entries_bar[:, plan.update_live] = V_bar[:, plan.update_positions[plan.update_live]]

    s = plan.selection
    carrier_bar += w_bar[:n] * s[:n] + np.conj(w_bar[n : 2 * n]) * s[n : 2 * n]
    lam_c = _host_splu(carrier_matrix).solve(carrier_bar, trans="T")
    r, c = plan.cmi
    carrier_entries_bar = np.where(c == n, lam_c[r], -lam_c[r] * carrier[np.minimum(c, n - 1)])
    carrier_entries_bar = np.where(plan.carrier_live, carrier_entries_bar, 0)
    return carrier_entries_bar, signal_entries_bar, noise_entries_bar


# ---------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------

# Arrays the host callbacks need (the remaining `simulate` arguments only select the path).
HOST_KEYS = (
    "optimized_parameter_indices",
    "optimized_value_indices",
    "carrier_changing_parameter_indices",
    "carrier_changing_values",
    "signal_changing_parameter_indices",
    "signal_changing_values",
    "parameters",
    "carrier_matrix",
    "signal_matrix",
    "noise_matrix",
    "noise_selection_vectors",
    "carrier_function_input_indices",
    "carrier_function_indices",
    "carrier_output_indices",
    "carrier_matrix_indices",
    "signal_function_input_indices",
    "signal_function_indices",
    "signal_output_indices",
    "signal_matrix_indices",
    "signal_carrier_indices",
    "signal_carrier_placing_indices",
    "noise_function_input_indices",
    "noise_function_indices",
    "noise_output_indices",
    "noise_system_matrix_indices",
)


def supports(arrays):
    """Static (shape/dtype only) check whether the fast path covers these `simulate` arguments."""
    optimized = arrays["optimized_parameters"]
    return (
        arrays["linked_indices"].shape[0] == 0
        and arrays["qhd_parameter_indices"].shape[0] == 0
        and arrays["noise_selection_vectors"].shape[0] == 1
        and arrays["carrier_changing_values"].ndim == 2
        and arrays["carrier_changing_values"].shape[1] == 1
        and arrays["signal_changing_values"].ndim == 2
        and optimized.ndim == 1
        and not jnp.issubdtype(optimized.dtype, jnp.complexfloating)
    )


def _output_shapes(arrays):
    n = arrays["carrier_matrix"].shape[0]
    m = arrays["signal_matrix"].shape[0]
    F = arrays["signal_changing_values"].shape[1]
    noise_shape = (F,) if F != 1 else ()  # `noise.squeeze()` of shape (1, F)
    return n, m, F, noise_shape


def _host_forward(optimized_parameters, *host_arrays):
    A = dict(zip(HOST_KEYS, host_arrays))
    entries = EntryPlan(A).values(optimized_parameters)  # derivatives are only taped in the backward pass
    carrier, X, noise, Y, MU, Z = forward(SystemPlan(A), *entries)
    return carrier[:, None], np.ascontiguousarray(X.T), noise.reshape(-1), Y, MU, Z


def _host_backward(optimized_parameters, *rest):
    k = len(HOST_KEYS)
    A = dict(zip(HOST_KEYS, rest[:k]))
    carrier, signal, noise, Y, MU, Z, carrier_bar, signal_bar, noise_bar = rest[k:]
    F = Y.shape[0]
    entry_plan = EntryPlan(A)
    (carrier_entries, signal_entries, _), tape = entry_plan.entries(optimized_parameters)
    entries_bar = backward(
        SystemPlan(A),
        carrier_entries,
        signal_entries,
        np.asarray(carrier)[:, 0],
        np.asarray(signal).T,
        np.asarray(noise).reshape(F),
        np.asarray(Y),
        np.asarray(MU),
        np.asarray(Z),
        np.asarray(carrier_bar)[:, 0],
        np.asarray(signal_bar).T,
        np.asarray(noise_bar, float).reshape(F),
    )
    gradient = entry_plan.pullback(tape, *entries_bar, value_count=optimized_parameters.shape[0])
    return gradient.astype(optimized_parameters.dtype)


def _forward_callback(optimized_parameters, host):
    n, m, F, noise_shape = _output_shapes(host)
    result_shapes = (
        jax.ShapeDtypeStruct((n, 1), jnp.complex128),
        jax.ShapeDtypeStruct((m, F), jnp.complex128),
        jax.ShapeDtypeStruct((F,), jnp.float64),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m, DETECTOR_WIDTH), jnp.complex128),
    )
    carrier, signal, noise, Y, MU, Z = jax.pure_callback(
        _host_forward, result_shapes, optimized_parameters, *(host[k] for k in HOST_KEYS), vmap_method="sequential"
    )
    return (carrier, signal, noise.reshape(noise_shape)), (Y, MU, Z)


def _is_primal(x):
    return isinstance(x, jax.custom_derivatives.CustomVJPPrimal)


def _zeros_if_symbolic(value):
    if isinstance(value, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(value.shape, value.dtype)
    return value


@jax.custom_vjp
def _fast(optimized_parameters, host, others):
    outputs, _ = _forward_callback(optimized_parameters, host)
    return outputs


def _fast_fwd(optimized_parameters, host, others):
    leaves = jax.tree_util.tree_leaves((host, others), is_leaf=_is_primal)
    unwrap = lambda tree: jax.tree_util.tree_map(lambda x: x.value, tree, is_leaf=_is_primal)
    p, host, others = optimized_parameters.value, unwrap(host), unwrap(others)
    if any(leaf.perturbed for leaf in leaves):
        # Differentiation with respect to something other than the optimized parameters:
        # defer to the original JAX implementation and its pullback.
        from differometor.simulate import simulate_jax

        full = lambda p, host, others: simulate_jax(optimized_parameters=p, **host, **others)
        outputs, pullback = jax.vjp(full, p, host, others)
        return outputs, (pullback,)
    outputs, (Y, MU, Z) = _forward_callback(p, host)
    return outputs, (p, host, outputs, Y, MU, Z)


def _fast_bwd(residuals, cotangents):
    cotangents = tuple(_zeros_if_symbolic(c) for c in cotangents)
    if len(residuals) == 1:
        return residuals[0](cotangents)
    p, host, (carrier, signal, noise), Y, MU, Z = residuals
    gradient = jax.pure_callback(
        _host_backward,
        jax.ShapeDtypeStruct(p.shape, p.dtype),
        p,
        *(host[k] for k in HOST_KEYS),
        carrier,
        signal,
        noise,
        Y,
        MU,
        Z,
        *cotangents,
        vmap_method="sequential",
    )
    return gradient, None, None


_fast.defvjp(_fast_fwd, _fast_bwd, symbolic_zeros=True)


def simulate_fast(arrays):
    """Fast `simulate` for argument dicts accepted by `supports`."""
    host = {key: arrays[key] for key in HOST_KEYS}
    others = {key: value for key, value in arrays.items() if key not in HOST_KEYS and key != "optimized_parameters"}
    return _fast(arrays["optimized_parameters"], host, others)
