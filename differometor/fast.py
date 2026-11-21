"""
Fast host-side simulation path: sparse LU solves with SuperLU and a hand-written adjoint.

`differometor.simulate.simulate` dispatches here for setups with exactly one quantum noise
detector (a `qnoised` or a balanced homodyne detector `qhd`) and a single carrier operating
point. Parameter links are covered for the linking functions listed in `LINK_RULES`. Everything
else uses the original JAX implementation (see `supports`).

How it works
------------
- Matrix entries are produced by the unchanged component functions of
  `differometor.components`, re-bound to a tiny NumPy shim. The forward pass evaluates plain
  values. The reverse pass evaluates them on dual numbers (value + derivative), but only for the
  function inputs that are design variables. No XLA compilation and no hand-written derivatives
  of the physics formulas are needed.
- The carrier system and the signal system of every frequency are assembled as sparse
  CSC matrices and factorized with SuperLU, the solver behind `scipy.sparse.linalg.splu`.
  The compiled SuperLU module is loaded directly (`_load_superlu`), because importing
  `scipy.sparse.linalg` costs more time than the whole simulation.
- All frequencies share one sparsity pattern. The scaling and the elimination order are found
  once (see `SystemPlan._analyze`) and reused. The frequencies are processed in blocks of
  BLOCK_FREQUENCIES: a block is one block-diagonal system for SuperLU (`_BlockLU`), with one
  factorization call and two solve calls.
- The quantum noise of the (single) detector is `Re(y^H N y)` with `y = S^-H w`; the
  same solve is reused as the adjoint helper for the gradient. A balanced homodyne detector
  reads two ports, the second one with a phase factor from its `phase` parameter.
- Reverse mode is written by hand. Cotangents follow JAX's convention for complex
  values (bilinear pairing), so signal adjoints solve with `S^-T`, not `S^-H`.

The JAX integration (`simulate_fast`) is a `jax.custom_vjp` around two
`jax.pure_callback`s. The callbacks keep no Python state: the forward callback returns
every residual the backward pass needs as arrays that flow through JAX, and the backward
callback rebuilds the sparsity plan from them.

Memory decides much of the run time here. JAX copies every array it hands to a callback and
every residual that a compiled program passes through, and memory that a callback touches for
the first time costs a page fault per 4 kB. Therefore:

- The constant system matrices arrive in coordinate form (`SparseMatrix`, made by the build
  step). As dense arrays the signal and the noise matrix of the benchmark setup take 9 MB each,
  for 1539 and 0 entries.
- Both passes work on one block of frequencies at a time, which keeps the intermediate arrays
  small, and the component functions write their outputs straight into the matrix entries.
- Every array that crosses a callback or program boundary also has a fixed cost (lowering,
  compilation, one copy per call). The index arrays of a setup are therefore static
  (`StaticArray`): JAX does not trace them and the callbacks get them directly. The remaining
  small arrays the reverse pass needs travel as one packed residual (`_pack`, `_unpack`).
"""

import functools
import importlib.machinery
import importlib.util
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import differometor.components as components
from differometor.components import UNIT_VACUUM, H_PLANCK, F0

NOISE_SCALE = UNIT_VACUUM * H_PLANCK * F0 * 0.25


class StaticArray(np.ndarray):
    """
    An integer array that belongs to the structure of a setup, not to its data: which parameter
    feeds which function, which matrix entry a result goes to. The build step returns its index
    arrays like this.

    For JAX a StaticArray is a pytree node without leaves. A traced function does not get a
    tracer for it but the array itself, so it is a constant of the compiled program, it never has
    to be copied into a host callback, and it costs nothing per call. Different index arrays give
    different compilations, as different static arguments do. Everywhere else it behaves like the
    NumPy array it is (read-only); `jnp.asarray` turns it into an ordinary JAX array.
    """

    def __new__(cls, values, dtype=int):
        array = np.array(values, dtype=dtype).view(cls)
        array.setflags(write=False)
        return array


class _StaticKey:
    """Hashable stand-in for a StaticArray in a pytree definition; equal contents compare equal."""

    __slots__ = ("array", "signature")

    def __init__(self, array):
        self.array, self.signature = array, hash((array.shape, array.dtype.str))

    def __hash__(self):
        return self.signature

    def __eq__(self, other):
        if not isinstance(other, _StaticKey):
            return NotImplemented
        a, b = self.array, other.array
        return a is b or (a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(np.asarray(a), np.asarray(b))))

    def __repr__(self):
        return f"StaticArray{self.array.shape}"


def _static_key(array):
    key = array.__dict__.get("_key")
    if key is None:
        key = array.__dict__["_key"] = _StaticKey(array)
    return key


jax.tree_util.register_pytree_node(StaticArray, lambda array: ((), _static_key(array)), lambda key, _: key.array)


class _Constant:
    """Static Python data inside a pytree (a node without leaves), for the residuals of `_fast`."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


jax.tree_util.register_pytree_node(_Constant, lambda constant: ((), constant.value), lambda value, _: _Constant(value))


class SparseMatrix:
    """
    A constant system matrix in coordinate form: `indices` (2, k) holds rows and columns of the
    nonzero entries, `values` (k,) their values, `shape` is the shape of the dense matrix.

    The build step hands the carrier, signal and noise matrix to `simulate` like this, because
    they are almost empty. The class is a JAX pytree with the children `indices` (a StaticArray
    from the build step) and `values`; the shape is static. `dense()` gives the dense JAX array,
    `np.asarray(matrix)` the dense NumPy array.
    """

    __slots__ = ("indices", "values", "shape")

    def __init__(self, indices, values, shape):
        self.indices, self.values, self.shape = indices, values, tuple(shape)

    @classmethod
    def from_dense(cls, matrix):
        """The coordinate form of a concrete dense matrix (entries in row-major order)."""
        matrix = np.asarray(matrix)
        rows, columns = np.nonzero(matrix)
        return cls(StaticArray([rows, columns]), jnp.asarray(matrix[rows, columns]), matrix.shape)

    def dense(self):
        rows, columns = np.asarray(self.indices) if isinstance(self.indices, StaticArray) else self.indices
        return jnp.zeros(self.shape, self.values.dtype).at[rows, columns].set(self.values)

    def __array__(self, dtype=None, copy=None):
        indices, values = np.asarray(self.indices), np.asarray(self.values)
        matrix = np.zeros(self.shape, values.dtype if dtype is None else dtype)
        matrix[indices[0], indices[1]] = values
        return matrix

    def __repr__(self):
        return f"SparseMatrix(shape={self.shape}, entries={self.values.shape[0]})"


jax.tree_util.register_pytree_node(
    SparseMatrix,
    lambda matrix: ((matrix.indices, matrix.values), matrix.shape),
    lambda shape, leaves: SparseMatrix(*leaves, shape),
)


# ---------------------------------------------------------------------
# SuperLU without the SciPy package import
# ---------------------------------------------------------------------

_SUPERLU_NAME = "scipy.sparse.linalg._dsolve._superlu"


def _load_superlu():
    """
    The SuperLU extension module that backs `scipy.sparse.linalg.splu`.

    Importing `scipy.sparse.linalg` executes a few hundred Python modules (about 90 ms) although
    only this one compiled module is used here. It is therefore loaded from its file. The module
    is registered under its regular name, so a later `import scipy.sparse.linalg` reuses it. If
    the file cannot be found, the regular import is used.
    """
    if _SUPERLU_NAME in sys.modules:
        return sys.modules[_SUPERLU_NAME]
    try:
        root = importlib.util.find_spec("scipy").submodule_search_locations[0]
        folder = os.path.join(root, "sparse", "linalg", "_dsolve")
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = os.path.join(folder, "_superlu" + suffix)
            if os.path.exists(path):
                loader = importlib.machinery.ExtensionFileLoader(_SUPERLU_NAME, path)
                spec = importlib.util.spec_from_loader(_SUPERLU_NAME, loader, origin=path)
                module = importlib.util.module_from_spec(spec)
                loader.exec_module(module)
                sys.modules[_SUPERLU_NAME] = module
                return module
    except Exception:
        pass
    from scipy.sparse.linalg._dsolve import _superlu

    return _superlu


_superlu = _load_superlu()


def _splu(size, values, row_indices, column_pointers, column_order="COLAMD", small_supernodes=False, diagonal_threshold=None):
    """
    LU factorization of a square CSC matrix, with the options `scipy.sparse.linalg.splu` passes.

    `column_order="NATURAL"` keeps the given column order (like `splu(..., permc_spec="NATURAL")`).

    `small_supernodes` sets SuperLU's `Relax` and `PanelSize` to 1. The systems here have about
    3.5 entries per row. With the default relaxed supernodes SuperLU pads the factors with explicit
    zeros, which makes both the factorization and every solve slower. Pivoting is unchanged.

    `diagonal_threshold` is SuperLU's `DiagPivotThresh` (`splu(..., diag_pivot_thresh=u)`): the
    diagonal entry is the pivot whenever its magnitude is at least `u` times the largest one in
    its column, otherwise the largest entry is. The default is 1, plain partial pivoting.
    """
    options = dict(DiagPivotThresh=diagonal_threshold, ColPerm=column_order, PanelSize=None, Relax=None)
    if column_order == "NATURAL":
        options["SymmetricMode"] = True
    if small_supernodes:
        options["PanelSize"] = 1
        options["Relax"] = 1
    return _superlu.gstrf(
        size,
        row_indices.size,
        np.ascontiguousarray(values, dtype=np.complex128),
        row_indices,
        column_pointers,
        csc_construct_func=None,
        ilu=False,
        options=options,
    )


# Columns of the detector basis Z = S^-T [e_d] kept per frequency: upper + lower sideband of one
# port. A balanced homodyne detector reads two ports and gets two more (`_detector_width`).
DETECTOR_WIDTH = 2

# Threshold pivoting for the signal systems (see `SystemPlan._analyze`).
DIAGONAL_PIVOT_THRESHOLD = 0.1
BALANCE_SWEEPS = 6

# Frequencies whose signal systems share one SuperLU call (see `_BlockLU`).
BLOCK_FREQUENCIES = 16


class _BlockLU:
    """LU factorizations of several square matrices that share one sparsity pattern.

    SuperLU has a fixed cost per call: work arrays, the elimination tree and a generous memory
    reserve for the factors. The matrices of BLOCK_FREQUENCIES frequencies are therefore stacked
    into one block-diagonal system, which takes one factorization call and one call per solve.
    No entry connects two blocks, so every pivot and every result is exactly the one of a
    separate factorization.

    `values` has one row per matrix, in the CSC order given by `row_indices`/`column_pointers`.
    """

    def __init__(self, size, values, row_indices, column_pointers, **options):
        entries = values.shape[1]
        self.size, self.blocks = size, []
        for start in range(0, values.shape[0], BLOCK_FREQUENCIES):
            chunk = values[start : start + BLOCK_FREQUENCIES]
            count = chunk.shape[0]
            offsets = np.arange(count, dtype=np.intc)[:, None]
            rows = (row_indices + offsets * size).ravel()
            pointers = np.empty(count * size + 1, np.intc)
            pointers[:-1] = (column_pointers[:-1] + offsets * entries).ravel()
            pointers[-1] = count * entries
            self.blocks.append((start, count, _splu(count * size, chunk.reshape(-1), rows, pointers, **options)))

    def solve(self, b, trans="N"):
        """b: (matrices, size, k) right-hand sides -> solutions of the same shape."""
        x = np.empty(b.shape, complex)
        for start, count, lu in self.blocks:
            block = b[start : start + count].reshape(count * self.size, -1)
            x[start : start + count] = lu.solve(block, trans=trans).reshape(count, self.size, -1)
        return x


def _balance(magnitudes, rows, cols, size, sweeps=BALANCE_SWEEPS):
    """
    log2 of a diagonal scaling d such that D^-1 A D has off-diagonal row and column maxima of
    similar size (matrix balancing on the entry magnitudes, as before an eigenvalue computation).

    `rows`/`cols` are the positions of `magnitudes` and have to be sorted by column. The scaling
    is a power of two per unknown, so applying it is exact. It leaves the diagonal untouched.
    Damped simultaneous sweeps: an unknown is rescaled by half of the square root of the ratio of
    its largest row and column entry, which removes most of the imbalance within a few sweeps.
    """
    keep = (rows != cols) & (magnitudes > 0)
    rows, cols = rows[keep], cols[keep]
    log_d = np.zeros(size)
    if not rows.size:
        return log_d
    log_a = np.log2(magnitudes[keep])
    by_row = np.argsort(rows, kind="stable")
    row_slots, row_starts = np.unique(rows[by_row], return_index=True)
    col_slots, col_starts = np.unique(cols, return_index=True)
    has_row = np.zeros(size, bool)
    has_row[row_slots] = True
    weight = np.full(size, 0.5)
    weight[col_slots[has_row[col_slots]]] = 0.25  # unknowns with row and column entries: meet in the middle
    for _ in range(sweeps):
        scaled = log_a + log_d[cols] - log_d[rows]
        step = np.zeros(size)
        step[row_slots] = np.maximum.reduceat(scaled[by_row], row_starts)
        step[col_slots] -= np.maximum.reduceat(scaled, col_starts)
        log_d += weight * step
    return np.clip(np.round(log_d), -500, 500)


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


def _function_outputs(function_index, arguments, columns=()):
    """
    The four outputs of one component function. `arguments` has one array per function input,
    of shape (rows, calls) or (1, calls).

    Each output is what the function returns: a scalar, an array that broadcasts to (rows, calls),
    or a `Dual` of those. Only the inputs listed in `columns` carry a derivative; the others are
    plain values.
    """
    width = len(columns)
    if width:
        arguments = list(arguments)
        seed = np.eye(width)
        for position, j in enumerate(columns):
            arguments[j] = Dual(arguments[j], seed[position].reshape(width, 1, 1))
    return FUNCTIONS_NUMPY[function_index](arguments)


def _group_sum(values, order, starts):
    """Sum the columns of `values` (R, E) that share a target slot (`order`/`starts` from `_grouping`)."""
    return np.add.reduceat(values[:, order], starts, axis=1)


def _grouping(targets):
    """Sort order, group starts and the distinct target slots for a scatter-add to `targets`."""
    order = np.argsort(targets, kind="stable")
    slots, starts = np.unique(targets[order], return_index=True)
    return order, starts, slots


class _EntryBlock:
    """One group of matrix entries (carrier, signal or noise) produced by component functions.

    `changing` lists the parameter columns that differ between the rows (the frequency sweep).
    Calls that read none of them give the same entries in every row and are evaluated once.
    The other calls get only the inputs that change as (rows, calls) arrays and the rest as
    (1, calls), so every part of a formula that does not depend on the frequency is also
    evaluated once (NumPy broadcasting does the rest).

    `differentiable` marks the parameter slots that receive a design variable. The reverse pass
    only needs derivatives with respect to those, so every other function input is evaluated as a
    plain value (a mirror surface has 7 inputs, but only reflectivity and tuning are design
    variables; force terms have none).
    """

    def __init__(self, function_input_indices, function_indices, output_indices, changing=(), differentiable=None):
        self.inputs = np.asarray(function_input_indices)
        function_indices = np.asarray(function_indices)
        self.outputs = np.asarray(output_indices)
        changes = np.isin(self.inputs, np.asarray(changing, int))
        self.varying = changes.any(axis=1)
        if differentiable is None:
            needed = np.ones(self.inputs.shape, bool)
        else:
            needed = differentiable[self.inputs]
        self.groups = []  # (function index, members, varying, differentiated input columns)
        self.changes = []  # per group: which function inputs differ between the rows
        for k in np.unique(function_indices):
            for varying in (True, False):
                members = np.nonzero((function_indices == k) & (self.varying == varying))[0]
                if members.size:
                    columns = np.nonzero(needed[members].any(axis=0))[0]
                    self.groups.append((int(k), members, varying, columns))
                    self.changes.append(changes[members].any(axis=0))

        # For every group and each of its four outputs: the entries that output fills (`slots`) and
        # the calls of the group they come from (`picks`). The function outputs are written straight
        # into the entries, without an intermediate (rows, calls, 4) array.
        calls = self.inputs.shape[0]
        call_group, call_place = np.empty(calls, np.intp), np.empty(calls, np.intp)
        for index, (_, members, _, _) in enumerate(self.groups):
            call_group[members] = index
            call_place[members] = np.arange(members.size)
        key = call_group[self.outputs[0]] * 4 + self.outputs[1]
        order = np.argsort(key, kind="stable")
        bounds = np.searchsorted(key[order], np.arange(len(self.groups) * 4 + 1))
        places = call_place[self.outputs[0][order]]
        self.writes = [
            [(order[bounds[i] : bounds[i + 1]], places[bounds[i] : bounds[i + 1]]) for i in range(4 * index, 4 * index + 4)]
            for index in range(len(self.groups))
        ]

    def _arguments(self, parameters, index):
        """One array per function input of a group, (R, calls) or (1, calls), and the shape they broadcast to."""
        _, members, varying, _ = self.groups[index]
        indices = self.inputs[members]
        first = parameters[:1]
        arguments = [(parameters if changes else first)[:, indices[:, j]] for j, changes in enumerate(self.changes[index])]
        return arguments, (parameters.shape[0] if varying else 1, members.size)

    @staticmethod
    def _write(entries, write, value, lead):
        """Put one function output (anything that broadcasts to `lead` = (rows or 1, calls)) into its entries."""
        slots, picks = write
        if slots.size:
            entries[:, slots] = np.broadcast_to(value, lead)[:, picks]

    def values(self, parameters):
        """parameters: (R, P) -> entries (R, E), no derivatives."""
        entries = np.empty((parameters.shape[0], self.outputs.shape[1]), complex)
        for index, ((function_index, _, _, _), writes) in enumerate(zip(self.groups, self.writes)):
            arguments, lead = self._arguments(parameters, index)
            for item, write in zip(_function_outputs(function_index, arguments), writes):
                self._write(entries, write, item, lead)
        return entries

    def run(self, parameters):
        """
        parameters: (R, P). Returns entries (R, E) and the tape for the reverse pass.

        The tape lists, per group, the function outputs that depend on a differentiated input:
        pairs (output index, tangent) with a tangent of shape (differentiated inputs, R or 1,
        calls). A tangent that is the same in every row keeps a single row.
        """
        entries = np.empty((parameters.shape[0], self.outputs.shape[1]), complex)
        tape = []
        for index, ((function_index, _, _, columns), writes) in enumerate(zip(self.groups, self.writes)):
            arguments, lead = self._arguments(parameters, index)
            derivatives = []
            for o, item in enumerate(_function_outputs(function_index, arguments, columns)):
                if isinstance(item, Dual):
                    if not np.isscalar(item.tan):
                        rows = item.tan.shape[-2] if item.tan.ndim > 1 else 1
                        derivatives.append((o, np.broadcast_to(item.tan, (columns.size, rows, lead[1]))))
                    item = item.val
                self._write(entries, writes[o], item, lead)
            tape.append(derivatives)
        return entries, tape

    def back(self, tape, entries_bar, parameter_count):
        """entries_bar: (R, E) -> real parameter cotangent summed over the rows (P,)."""
        rows, calls = entries_bar.shape[0], self.inputs.shape[0]
        # Scatter from entries to (call, output) slots; several entries may share a slot.
        order, starts, slots = _grouping(self.outputs[0] * 4 + self.outputs[1])
        summed = _group_sum(entries_bar, order, starts) if order.size else None
        slot_calls, slot_outputs = slots // 4, slots % 4
        position = np.empty(calls, np.intp)
        parameters_bar = np.zeros(parameter_count)
        for (function_index, members, varying, columns), derivatives in zip(self.groups, tape):
            if not derivatives:
                continue
            # Output cotangents of this group's calls only: the scatter slots that belong to them
            # (`picked`) and their place in a (members, 4) block (`local`).
            position.fill(-1)
            position[members] = np.arange(members.size)
            picked = np.nonzero(position[slot_calls] >= 0)[0]
            local = position[slot_calls[picked]] * 4 + slot_outputs[picked]
            if varying:
                outputs_bar = np.zeros((rows, members.size * 4), complex)
                if picked.size:
                    outputs_bar[:, local] = summed[:, picked]
            else:  # identical outputs in every row: only the sum over the rows matters
                outputs_bar = np.zeros((1, members.size * 4), complex)
                if picked.size:
                    outputs_bar[0, local] = summed[:, picked].sum(0)
            outputs_bar = outputs_bar.reshape(-1, members.size, 4)
            inputs_bar = np.zeros((members.size, columns.size))
            for o, tangent in derivatives:
                bar = outputs_bar[:, :, o]
                if tangent.shape[1] == 1:  # same derivative in every row
                    inputs_bar += (bar.sum(0) * tangent[:, 0]).real.T
                else:
                    inputs_bar += np.einsum("rc,jrc->cj", bar, tangent).real
            parameters_bar += np.bincount(self.inputs[members][:, columns].ravel(), inputs_bar.ravel(), parameter_count)
        return parameters_bar


# Linking functions (derived parameters, e.g. a signal amplitude that follows `sqrt(laser power)`)
# the fast path evaluates on the host: the NumPy value and the derivative as a function of input
# and value. Setups with any other linking function use the original JAX implementation.
LINK_RULES = ((jnp.sqrt, np.sqrt, lambda x, y: 0.5 / y),)


def _link_rule(function):
    """(value, derivative) for a linking function of `LINK_RULES`, None for any other function."""
    for known, value, derivative in LINK_RULES:
        if function is known:
            return value, derivative
    return None


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
        # Parameter links: slot `link_targets[i]` gets `rule(slot link_sources[i])`.
        self.link_sources = np.asarray(A["linked_indices"], np.intp).reshape(-1)
        self.link_targets = np.asarray(A["indices_to_link"], np.intp).reshape(-1)
        self.link_rules = []
        if self.link_sources.size:
            from differometor.build import LINKING_FUNCTIONS

            self.link_rules = [_link_rule(LINKING_FUNCTIONS[i]) for i in np.asarray(A["linking_function_indices"]).reshape(-1)]
        # Balanced homodyne detector: the selection entries `qhd_places` carry a phase factor.
        self.qhd_parameter_indices = np.asarray(A["qhd_parameter_indices"], np.intp).reshape(-1)
        self.qhd_places = np.asarray(A["qhd_placing_indices"], np.intp).reshape(-1, 3)[:, 2]
        self.selection_base = np.asarray(A["noise_selection_vectors"])[0, 0]
        # Parameter slots whose derivative reaches a design variable (see `pullback`).
        differentiable = np.zeros(self.base.size, bool)
        differentiable[self.optimized_parameter_indices] = True
        differentiable[self.link_targets] = differentiable[self.link_sources]
        differentiable[self.carrier_changing_indices] = False
        differentiable[self.signal_changing_indices] = False
        block = lambda name, changing=(): _EntryBlock(
            A[name + "_function_input_indices"], A[name + "_function_indices"], A[name + "_output_indices"], changing, differentiable
        )
        self.carrier, self.signal, self.noise = block("carrier"), block("signal", changing), block("noise", changing)

    def _design_row(self, optimized_parameters):
        """The parameter row with the design variables in place, before the links."""
        P = self.base.copy()
        P[self.optimized_parameter_indices] = np.asarray(optimized_parameters, float)[self.optimized_value_indices]
        return P

    def _row(self, optimized_parameters):
        """The parameter row with design variables and linked parameters, before the sweeps."""
        P = self._design_row(optimized_parameters)
        if self.link_targets.size:
            # All links read the row as it is before any of them writes (as in `simulate_jax`).
            sources = P[self.link_sources]
            P[self.link_targets] = [value(x) for (value, _), x in zip(self.link_rules, sources)]
        return P

    def selection(self, optimized_parameters):
        """
        The selection vector of the noise detector. A balanced homodyne detector multiplies the
        entries of its second port by `exp(1j * phase)`, with the phase parameter in degrees.
        """
        selection = self.selection_base
        if self.qhd_parameter_indices.size:
            phases = np.exp(1j * np.radians(self._row(optimized_parameters)[self.qhd_parameter_indices]))
            selection = selection.astype(complex)
            np.multiply.at(selection, self.qhd_places, phases)
        return selection

    def _rows(self, optimized_parameters):
        P = self._row(optimized_parameters)
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

    def pullback(self, tape, carrier_bar, signal_bar, noise_bar, value_count, optimized_parameters=None, selection_bar=None):
        """
        The gradient with respect to the design variables from the cotangents of the entries.

        `optimized_parameters` is needed when the setup has parameter links, `selection_bar` (the
        cotangent of `selection`) when it has a balanced homodyne detector.
        """
        carrier_tape, signal_tape, noise_tape = tape
        parameter_count = self.base.size
        signal_rows_bar = self.signal.back(signal_tape, signal_bar, parameter_count) + self.noise.back(
            noise_tape, noise_bar, parameter_count
        )
        signal_rows_bar[self.signal_changing_indices] = 0.0  # overwritten by the sweep values
        P_bar = self.carrier.back(carrier_tape, carrier_bar[None], parameter_count) + signal_rows_bar
        P_bar[self.carrier_changing_indices] = 0.0
        if self.qhd_parameter_indices.size and selection_bar is not None:
            # selection = base * exp(1j * radians(phase)): d selection / d phase = 1j * pi / 180 * selection
            places = self.qhd_places
            phase_bar = (selection_bar[places] * 1j * self.selection(optimized_parameters)[places]).real * (np.pi / 180)
            np.add.at(P_bar, self.qhd_parameter_indices, phase_bar)
        if self.link_targets.size:
            sources = self._design_row(optimized_parameters)[self.link_sources]
            targets_bar = P_bar[self.link_targets]
            P_bar[self.link_targets] = 0.0  # overwritten by the links
            with np.errstate(divide="ignore", invalid="ignore"):
                slopes = np.array([derivative(x, value(x)) for (value, derivative), x in zip(self.link_rules, sources)])
                np.add.at(P_bar, self.link_sources, slopes * targets_bar)
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
    """Static sparsity data derived from the index arrays (cheap, rebuilt in every callback).

    `A` holds the host arrays (HOST_KEYS), `n` is the size of the carrier system. The forward
    pass needs the complete sparsity pattern (`signal_base=True`, `noise_base=True`). The reverse
    pass only needs the entries that depend on parameters: the ones written by component
    functions or multiplied by the carrier. `signal_base=False` restricts the signal pattern to
    those entries and `noise_base=False` leaves out the constant noise entries.

    `selection` is the selection vector of the noise detector (`EntryPlan.selection`; by default
    the stored one, which is right without a balanced homodyne detector). `width` is the number
    of columns the detector basis Z has (`_detector_width`).
    """

    def __init__(self, A, n, signal_base=True, noise_base=True, selection=None, width=DETECTOR_WIDTH):
        get = lambda key: np.asarray(A[key])
        self.n = n
        self.selection = get("noise_selection_vectors")[0, 0] if selection is None else selection
        self.m = m = self.selection.size

        # Carrier system: the n x n matrix in CSC order, followed by the right-hand side (column
        # n). Entries are identified by the key `column * n + row`, which sorts them that way.
        self.cmi = cmi = get("carrier_matrix_indices") % np.array([[n], [n + 1]])
        self.carrier_live = _last_writer(cmi[0], cmi[1], n + 1)
        base_rows, base_cols = get("carrier_base_indices")
        base_keys, written_keys = base_cols * n + base_rows, cmi[1] * n + cmi[0]
        keys = np.unique(np.concatenate([base_keys, written_keys]))
        self.carrier_count = count = int(np.searchsorted(keys, n * n))  # entries of the matrix itself
        self.carrier_rows = (keys[:count] % n).astype(np.intc)
        self.carrier_ptr = np.searchsorted(keys[:count], np.arange(n + 1) * n).astype(np.intc)
        self.carrier_rhs_rows = keys[count:] - n * n
        self.carrier_base = np.zeros(keys.size, complex)
        self.carrier_base[np.searchsorted(keys, base_keys)] = get("carrier_base_values")
        self.carrier_positions = np.searchsorted(keys, written_keys)

        # Signal system: pattern of the augmented m x (m + 1) matrix in CSC order. Entries are
        # identified by the key `column * m + row`, which sorts them column by column.
        smi = get("signal_matrix_indices") % np.array([[m], [m + 1]])
        placing = get("signal_carrier_placing_indices") % np.array([[m], [m + 1]])
        self.signal_carrier_indices = get("signal_carrier_indices").ravel()
        key = lambda rows, cols: cols * m + rows
        written = [key(smi[0], smi[1]), key(placing[0], placing[1])]
        base_keys = key(*get("signal_base_indices"))
        if signal_base:
            written.append(base_keys)
        keys = np.unique(np.concatenate(written))
        cols, rows = np.divmod(keys, m)
        self.rows, self.cols, self.size = rows, cols, rows.size
        # Constant values on the pattern. Without `signal_base` the pattern holds only some of the
        # constant entries, among them the ones in placing slots, which the reverse pass needs.
        self.signal_base = np.zeros(keys.size, complex)
        if keys.size:
            places = np.minimum(np.searchsorted(keys, base_keys), keys.size - 1)
            present = keys[places] == base_keys
            self.signal_base[places[present]] = get("signal_base_values")[present]
        self.update_positions = np.searchsorted(keys, key(smi[0], smi[1]))
        self.update_live = _last_writer(smi[0], smi[1], m + 1)
        self.placing_positions = np.searchsorted(keys, key(placing[0], placing[1]))
        self.placing_live = _last_writer(placing[0], placing[1], m + 1)
        # Entries that are complex-conjugated for the lower sideband (see `simulate`).
        self.conjugated = (
            ((rows >= n) & (rows < 2 * n) & (cols >= n) & (cols < 2 * n))
            | ((rows >= 2 * n) & (cols < n))
            | ((rows >= n) & (rows < 2 * n) & (cols >= 2 * n))
        )
        self.in_matrix = cols < m
        self.matrix_rows = rows[self.in_matrix].astype(np.intc)
        self.matrix_cols = cols[self.in_matrix]
        self.matrix_ptr = np.searchsorted(self.matrix_cols, np.arange(m + 1)).astype(np.intc)
        self.rhs_rows = rows[~self.in_matrix]

        # Noise matrix: base entries plus entries written by noise component functions.
        nmi = get("noise_system_matrix_indices") % m
        self.noise_live = _last_writer(nmi[0], nmi[1], m)
        if noise_base:  # constant entries that no noise component function overwrites
            base_rows, base_cols = get("noise_base_indices")
            kept = ~np.isin(base_rows * m + base_cols, nmi[0] * m + nmi[1])
            base_rows, base_cols = base_rows[kept], base_cols[kept]
            self.noise_base = get("noise_base_values")[kept]
        else:
            base_rows = base_cols = np.empty(0, np.intp)
            self.noise_base = np.empty(0, complex)
        self.noise_rows = np.concatenate([base_rows, nmi[0][self.noise_live]])
        self.noise_cols = np.concatenate([base_cols, nmi[1][self.noise_live]])
        if noise_base:  # scatter plans for N y and N^H y (forward pass only)
            self.noise_by_row = _grouping(self.noise_rows)
            self.noise_by_col = _grouping(self.noise_cols)
        # Rows the quantum noise detector reads. When they fit into the detector basis, the forward
        # pass solves S^T z = e_d for each of them; Y and the signal adjoint are then combinations of z.
        self.detector = np.nonzero(self.selection)[0]
        self.width = width
        self.use_basis = 0 < self.detector.size <= width
        self.order = None

    def carrier_matrix(self, carrier_entries):
        """The factorized carrier matrix and the right-hand side of the carrier system."""
        count, live = self.carrier_count, self.carrier_live
        values = self.carrier_base.copy()
        values[self.carrier_positions[live]] = carrier_entries[live]
        rhs = np.zeros(self.n, complex)
        rhs[self.carrier_rhs_rows] = values[count:]
        return _splu(self.n, values[:count], self.carrier_rows, self.carrier_ptr), rhs

    def signal_values(self, signal_entries, carrier):
        """Nonzero values of the augmented signal matrices, shape (F, size)."""
        F = signal_entries.shape[0]
        V = np.empty((F, self.size), complex)
        V[:] = self.signal_base
        V[:, self.update_positions[self.update_live]] = signal_entries[:, self.update_live]
        placed = self.placing_positions[self.placing_live]
        V[:, placed] = -V[:, placed] * carrier[self.signal_carrier_indices[self.placing_live]]
        V[:, self.conjugated] = np.conj(V[:, self.conjugated])
        return V

    def placing_writers(self):
        """For the placing slots a signal entry writes to: the slots and the entries (last writer)."""
        writer = np.full(self.size, -1)
        writer[self.update_positions[self.update_live]] = np.nonzero(self.update_live)[0]
        writer = writer[self.placing_positions]
        written = np.nonzero(writer >= 0)[0]
        return written, writer[written]

    def _analyze(self, values_row):
        """
        Scaling and elimination order, found on one matrix and reused for all frequencies.

        Every equation of the signal system has the form x_i = sum_j m_ij x_j + b_i, so the
        diagonal is 1 and eliminating unknown i with its own equation is a substitution. Doing
        that in a minimum-degree order of the pattern of S + S^T gives far fewer factor entries
        than an order that has to be safe for arbitrary row exchanges (6100 instead of 8330
        for the benchmark setup). SuperLU still checks every pivot: the diagonal is only used
        while it is at least DIAGONAL_PIVOT_THRESHOLD times the largest entry of its column,
        otherwise the largest entry is taken (threshold partial pivoting).

        That test compares entries of one column, which only means something if the unknowns
        have comparable units. Here they do not (fields, forces and displacements: entries
        from 1e-11 to 1e13), so the system is first balanced with D^-1 S D.

        The factorized matrix is B = D^-1 S[order][:, order] D, with rows and columns in
        elimination order.
        """
        m = self.m
        rows, cols = self.matrix_rows, self.matrix_cols
        values = values_row[self.in_matrix]
        log_scale = _balance(np.abs(values), rows, cols, m)
        entry_scale = np.exp2(log_scale[cols] - log_scale[rows])
        first = _splu(
            m, values * entry_scale, rows, self.matrix_ptr, "MMD_AT_PLUS_A", True, DIAGONAL_PIVOT_THRESHOLD
        )
        position = np.asarray(first.perm_c, np.intp)  # unknown -> place in the elimination order
        new_rows, new_cols = position[rows], position[cols]
        layout = np.lexsort((new_rows, new_cols))
        self.ordered_rows = new_rows[layout].astype(np.intc)
        self.ordered_ptr = np.searchsorted(new_cols[layout], np.arange(m + 1)).astype(np.intc)
        self.ordered_values = np.nonzero(self.in_matrix)[0][layout]
        self.ordered_scale = entry_scale[layout]
        self.position = position
        self.order = np.argsort(position)  # place -> unknown
        self.scale = np.exp2(log_scale)  # d, per unknown

    def signal_factors(self, V):
        """SuperLU factorizations of B_f = D^-1 S_f[order][:, order] D for all frequencies (rows of `V`)."""
        if self.order is None:
            self._analyze(V[0])
        values = V[:, self.ordered_values]
        values *= self.ordered_scale
        return _BlockLU(
            self.m,
            values,
            self.ordered_rows,
            self.ordered_ptr,
            column_order="NATURAL",
            small_supernodes=True,
            diagonal_threshold=DIAGONAL_PIVOT_THRESHOLD,
        )

    def solve_transposed(self, factors, c):
        """lam_f with S_f^T lam_f = c_f for every frequency: B^T (D lam) = D c in elimination order."""
        order = self.order
        scale = self.scale[order]
        lam = np.empty(c.shape, complex)
        lam[:, order] = factors.solve((c[:, order] * scale)[:, :, None], trans="T")[:, :, 0] / scale
        return lam

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


def _scatter_sum(values, grouping, size):
    """Scatter-add the columns of `values` (F, E) into (F, size) target slots."""
    order, starts, slots = grouping
    result = np.zeros((values.shape[0], size), complex)
    if order.size:
        result[:, slots] = _group_sum(values, order, starts)
    return result


def forward(plan, carrier_entries, signal_entries, noise_entries):
    """Carrier, signal (m, F) and noise outputs plus the adjoint helpers Y = S^-H w, MU and Z.

    Z[f] = S_f^-T [e_d for d in plan.detector] (zero-padded to `plan.width` columns). It gives
    Y = conj(Z conj(w_d)) here and the signal adjoint S^-T signal_bar in `backward` without a
    second factorization, whenever the cotangent of the signal lives on the detector rows.

    The frequencies are processed in blocks of BLOCK_FREQUENCIES. A block takes one factorization
    call and two solve calls with two right-hand sides per frequency: first S^T Z = [e_d], then
    S [X, MU] = [rhs, (N + N^H) Y / 2]. Everything else is evaluated for the whole block at once.
    Working block by block keeps the intermediate arrays small: all of them together would be
    several times larger than the results, and freshly allocated memory is slow.

    SuperLU factorizes B = D^-1 S[order][:, order] D (see `SystemPlan._analyze`), so
    S x = b becomes B (D^-1 x) = D^-1 b and S^T z = c becomes B^T (D z) = D c, with all vectors in
    elimination order. The quantities between the two solves stay in that order (suffix `_o`).
    """
    m = plan.m
    carrier_lu, carrier_rhs = plan.carrier_matrix(carrier_entries)
    carrier = carrier_lu.solve(carrier_rhs)
    w = plan.detector_weights(carrier)
    F = signal_entries.shape[0]
    d, k = plan.detector, plan.detector.size
    signal = np.empty((m, F), complex)
    MU = np.empty((F, m), complex)
    Y = np.empty((F, m), complex)
    Z = np.zeros((F, m, plan.width), complex)
    power = np.empty(F)
    for start in range(0, F, BLOCK_FREQUENCIES):
        block = slice(start, start + BLOCK_FREQUENCIES)
        V = plan.signal_values(signal_entries[block], carrier)
        N = plan.noise_values(noise_entries[block])
        count = V.shape[0]
        if start == 0:
            if plan.order is None:
                plan._analyze(V[0])
            order, position = plan.order, plan.position
            scale = plan.scale[order]
            rhs_places, rhs_scale = position[plan.rhs_rows], plan.scale[plan.rhs_rows]
            by_row = (*plan.noise_by_row[:2], position[plan.noise_by_row[2]])
            by_col = (*plan.noise_by_col[:2], position[plan.noise_by_col[2]])
            noise_row_places, noise_col_places = position[plan.noise_rows], position[plan.noise_cols]
        factors = plan.signal_factors(V)
        if plan.use_basis:
            basis = np.zeros((count, m, k), complex)
            basis[:, position[d], np.arange(k)] = plan.scale[d]
            Z_o = factors.solve(basis, trans="T")
            Z_o /= scale[:, None]
            Y_o = np.conj(Z_o @ np.conj(w[d]))
            Z[block][:, order, :k] = Z_o
        else:
            w_o = np.empty((count, m, 1), complex)
            w_o[:, :, 0] = w[order] * scale
            Y_o = factors.solve(w_o, trans="H")[:, :, 0]
            Y_o /= scale
        q_o = _scatter_sum(N * Y_o[:, noise_col_places], by_row, m)  # N y
        qh_o = _scatter_sum(np.conj(N) * Y_o[:, noise_row_places], by_col, m)  # N^H y
        power[block] = np.einsum("fm,fm->f", np.conj(Y_o), q_o).real
        right_hand_sides = np.zeros((count, m, 2), complex)
        right_hand_sides[:, rhs_places, 0] = V[:, ~plan.in_matrix] / rhs_scale
        right_hand_sides[:, :, 1] = (q_o + qh_o) * (0.5 / scale)
        solutions = factors.solve(right_hand_sides)
        solutions *= scale[:, None]
        signal[order, block] = solutions[:, :, 0].T
        MU[block][:, order] = solutions[:, :, 1]
        Y[block][:, order] = Y_o
    noise = np.sqrt(2 * power * NOISE_SCALE)
    return carrier, signal, noise, Y, MU, Z


def _signal_adjoint(plan, Z, signal_bar, general_adjoint):
    """lam_f = S_f^-T signal_bar_f for every frequency, shape (F, m).

    The detector basis Z covers a cotangent that lives on the detector rows (the usual readout).
    Any other cotangent needs the complete signal systems again: `general_adjoint(signal_bar)`.
    """
    support = np.nonzero(np.any(signal_bar != 0, axis=0))[0]
    d, k = plan.detector, plan.detector.size
    if plan.use_basis and np.isin(support, d).all():
        return np.einsum("fmk,fk->fm", Z[:, :, :k], signal_bar[:, d])
    return general_adjoint(signal_bar)


def backward(
    plan,
    carrier_entries,
    signal_entries,
    carrier,
    X,
    noise,
    Y,
    MU,
    Z,
    carrier_bar,
    signal_bar,
    noise_bar,
    general_adjoint,
):
    """Cotangents of the carrier, signal and noise entries (JAX's bilinear convention).

    `plan` only has to contain the parameter-dependent entries (`SystemPlan(..., signal_base=False,
    noise_base=False)`): constant matrix entries receive no cotangent. `X` is the signal (F, m).
    """
    n, m = plan.n, plan.m
    carrier_lu, _ = plan.carrier_matrix(carrier_entries)
    F = signal_entries.shape[0]
    carrier_bar = carrier_bar.astype(complex)
    g = noise_bar * NOISE_SCALE / noise  # d noise / d power
    lam = _signal_adjoint(plan, Z, signal_bar, general_adjoint)
    in_matrix, off_matrix = np.nonzero(plan.in_matrix)[0], np.nonzero(~plan.in_matrix)[0]
    rows, cols = plan.rows[in_matrix], plan.cols[in_matrix]
    conjugated = np.nonzero(plan.conjugated)[0]
    live = np.nonzero(plan.placing_live)[0]
    placed = plan.placing_positions[live]
    sci = plan.signal_carrier_indices[live]
    update_live = np.nonzero(plan.update_live)[0]
    updated = plan.update_positions[update_live]
    noise_live = np.nonzero(plan.noise_live)[0]
    written, writers = plan.placing_writers()
    placing_base = plan.signal_base[plan.placing_positions]
    signal_entries_bar = np.zeros_like(signal_entries)
    noise_entries_bar = np.zeros((F, plan.noise_live.size), complex)
    placed_sum = np.zeros(placed.size, complex)
    # Frequency blocks keep the intermediate arrays small, as in `forward`.
    for start in range(0, F, BLOCK_FREQUENCIES):
        block = slice(start, start + BLOCK_FREQUENCIES)
        lam_b, Y_b, g_b = lam[block], Y[block], g[block, None]
        count = lam_b.shape[0]
        # -lam X - 2 g conj(Y) MU on the matrix pattern, evaluated in place.
        first = lam_b[:, rows]
        first *= X[block][:, cols]
        np.negative(first, out=first)
        second = Y_b[:, rows]
        np.conj(second, out=second)
        second *= 2 * g_b
        second *= MU[block][:, cols]
        first -= second
        V_bar = np.empty((count, plan.size), complex)
        V_bar[:, in_matrix] = first
        V_bar[:, off_matrix] = lam_b[:, plan.rhs_rows]

        N_bar = g_b * np.conj(Y_b[:, plan.noise_rows]) * Y_b[:, plan.noise_cols]
        noise_entries_bar[block][:, noise_live] = N_bar[:, plan.noise_base.size :]

        V_bar[:, conjugated] = np.conj(V_bar[:, conjugated])
        # Values of the placing slots before the placement: a signal entry or the constant base.
        pre = np.empty((count, plan.placing_positions.size), complex)
        pre[:] = placing_base
        pre[:, written] = signal_entries[block][:, writers]
        placed_bar = V_bar[:, placed]
        placed_sum += (placed_bar * pre[:, live]).sum(0)
        placed_bar *= carrier[sci]
        V_bar[:, placed] = np.negative(placed_bar, out=placed_bar)
        signal_entries_bar[block][:, update_live] = V_bar[:, updated]
    np.add.at(carrier_bar, sci, -placed_sum)
    w_bar = 2 * np.conj(g @ MU)  # sum over the frequencies of 2 g conj(MU); g is real

    s = plan.selection
    carrier_bar += w_bar[:n] * s[:n] + np.conj(w_bar[n : 2 * n]) * s[n : 2 * n]
    lam_c = carrier_lu.solve(carrier_bar, trans="T")
    r, c = plan.cmi
    carrier_entries_bar = np.where(c == n, lam_c[r], -lam_c[r] * carrier[np.minimum(c, n - 1)])
    carrier_entries_bar = np.where(plan.carrier_live, carrier_entries_bar, 0)
    return carrier_entries_bar, signal_entries_bar, noise_entries_bar


def selection_cotangent(carrier, noise, MU, noise_bar):
    """
    Cotangent of the selection vector of the noise detector, shape (m,). The detector weights are
    w = selection * carrier for the upper sideband, conj(selection * carrier) for the lower one
    and the selection itself elsewhere; their cotangent is the `w_bar` of `backward`.
    """
    n = carrier.shape[0]
    selection_bar = 2 * np.conj((noise_bar * NOISE_SCALE / noise) @ MU)
    selection_bar[:n] *= carrier
    selection_bar[n : 2 * n] = np.conj(selection_bar[n : 2 * n]) * carrier
    return selection_bar


# ---------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------

# `simulate` arguments the host callbacks need (the remaining ones only select the path): plain
# arrays and the three constant system matrices, which have to be `SparseMatrix` objects.
ARRAY_KEYS = (
    "optimized_parameter_indices",
    "optimized_value_indices",
    "carrier_changing_parameter_indices",
    "carrier_changing_values",
    "signal_changing_parameter_indices",
    "signal_changing_values",
    "parameters",
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
    "qhd_parameter_indices",
    "qhd_placing_indices",
    "linked_indices",
    "linking_function_indices",
    "indices_to_link",
)
MATRIX_KEYS = ("carrier_matrix", "signal_matrix", "noise_matrix")
# Names of the arrays a host callback works with: the plain arrays, then both leaves of every
# matrix. Index arrays that the build step marked as static (`StaticArray`) stay with the
# callbacks. The others go through JAX: the forward callback gets them as operands and packs them
# into at most two residuals for the reverse callback (`_pack`), because every array that crosses
# a callback or program boundary costs time, however small it is.
HOST_KEYS = ARRAY_KEYS + (
    "carrier_base_indices",
    "carrier_base_values",
    "signal_base_indices",
    "signal_base_values",
    "noise_base_indices",
    "noise_base_values",
)


def _host_leaves(host):
    """The arrays behind HOST_KEYS, in that order, taken from the `simulate` arguments."""
    matrices = [leaf for key in MATRIX_KEYS for leaf in (host[key].indices, host[key].values)]
    return [host[key] for key in ARRAY_KEYS] + matrices


def _split_host(host):
    """
    The host arrays in two groups: `static` {name: StaticArray} and the ones that travel through
    JAX, as their `layout` ((name, dtype character, shape) per array) and the arrays themselves.
    """
    static, layout, traced = {}, [], []
    for key, leaf in zip(HOST_KEYS, _host_leaves(host)):
        if isinstance(leaf, StaticArray):
            static[key] = leaf
        else:
            layout.append((key, np.dtype(leaf.dtype).char, tuple(leaf.shape)))
            traced.append(leaf)
    return static, tuple(layout), traced


def _is_integer(character):
    return np.dtype(character).kind in "iub"


def _packed_sizes(layout):
    """Lengths of the integer and of the real array that `_pack` makes for this layout."""
    integers = sum(math.prod(shape) for _, character, shape in layout if _is_integer(character))
    reals = sum(
        math.prod(shape) * (2 if np.dtype(character).kind == "c" else 1)
        for _, character, shape in layout
        if not _is_integer(character)
    )
    return integers, reals


def _pack(A, layout):
    """
    The arrays `A[name]` of `layout` as flat arrays; `_unpack` restores them exactly.

    Integer and boolean arrays go into one int64 array, real and complex arrays into one float64
    array, where a complex number takes two places. An array that would be empty is left out.
    """
    integers, reals = [], []
    for key, character, _ in layout:
        array = np.asarray(A[key])
        if _is_integer(character):
            integers.append(array.reshape(-1))
        elif array.dtype.kind == "c":
            reals.append(np.ascontiguousarray(array, np.complex128).reshape(-1).view(np.float64))
        else:
            reals.append(array.reshape(-1))
    packed = []
    if sum(part.size for part in integers):
        packed.append(np.concatenate(integers, dtype=np.int64))
    if sum(part.size for part in reals):
        packed.append(np.concatenate(reals, dtype=np.float64))
    return packed


def _unpack(layout, packed):
    """The dictionary of arrays that `_pack` packed, as views wherever the dtype allows it."""
    sizes = _packed_sizes(layout)
    packed = [np.asarray(array) for array in packed]
    packed_integers = packed.pop(0) if sizes[0] else None
    packed_reals = packed.pop(0) if sizes[1] else None
    A, integer_at, real_at = {}, 0, 0
    for key, character, shape in layout:
        dtype, size = np.dtype(character), math.prod(shape)
        if _is_integer(character):
            flat = packed_integers[integer_at : integer_at + size] if size else np.empty(0, np.int64)
            integer_at += size
        else:
            width = 2 if dtype.kind == "c" else 1
            flat = packed_reals[real_at : real_at + size * width] if size else np.empty(0, np.float64)
            real_at += size * width
            if width == 2:
                flat = flat.view(np.complex128)
        A[key] = flat.astype(dtype, copy=False).reshape(shape)
    return A


def _links_supported(arrays):
    """Parameter links are covered when their functions are known at trace time and in `LINK_RULES`."""
    if arrays["linked_indices"].shape[0] == 0:
        return True
    indices = arrays["linking_function_indices"]
    if not isinstance(indices, StaticArray):
        return False
    from differometor.build import LINKING_FUNCTIONS

    return all(0 <= i < len(LINKING_FUNCTIONS) and _link_rule(LINKING_FUNCTIONS[i]) is not None for i in indices.reshape(-1).tolist())


def supports(arrays):
    """
    Static check whether the fast path covers these `simulate` arguments (types, shapes, dtypes
    and the linking functions): one quantum noise detector, which may be a balanced homodyne
    detector, a single carrier operating point, real design variables and known parameter links.
    """
    optimized = arrays["optimized_parameters"]
    return (
        all(isinstance(arrays[key], SparseMatrix) for key in MATRIX_KEYS)
        and arrays["noise_selection_vectors"].shape[0] == 1
        and arrays["carrier_changing_values"].ndim == 2
        and arrays["carrier_changing_values"].shape[1] == 1
        and arrays["signal_changing_values"].ndim == 2
        and optimized.ndim == 1
        and not jnp.issubdtype(optimized.dtype, jnp.complexfloating)
        and _links_supported(arrays)
    )


def _detector_width(arrays):
    """
    Columns of the detector basis Z: the upper and the lower sideband of every port the noise
    detector reads. A `qnoised` reads one port, a balanced homodyne detector a second one, which
    has two rows in `qhd_placing_indices`.
    """
    return DETECTOR_WIDTH + math.prod(arrays["qhd_placing_indices"].shape) // 3


def _output_shapes(arrays):
    n = arrays["carrier_matrix"].shape[0]
    m = arrays["signal_matrix"].shape[0]
    F = arrays["signal_changing_values"].shape[1]
    noise_shape = (F,) if F != 1 else ()  # `noise.squeeze()` of shape (1, F)
    return n, m, F, noise_shape


def _host_forward(n, width, static, layout, optimized_parameters, *traced):
    A = dict(static)
    A.update((key, array) for (key, _, _), array in zip(layout, traced))
    entry_plan = EntryPlan(A)
    entries = entry_plan.values(optimized_parameters)  # derivatives are only taped in the backward pass
    plan = SystemPlan(A, n, selection=entry_plan.selection(optimized_parameters), width=width)
    carrier, signal, noise, Y, MU, Z = forward(plan, *entries)
    return carrier[:, None], signal, noise.reshape(-1), Y, MU, Z, *_pack(A, layout)


def _host_backward(static, layout, optimized_parameters, carrier, signal, noise, Y, MU, Z, *rest):
    carrier_bar, signal_bar, noise_bar = rest[-3:]
    A = dict(static)
    A.update(_unpack(layout, rest[:-3]))
    F, n = Y.shape[0], carrier.shape[0]
    entry_plan = EntryPlan(A)
    (carrier_entries, signal_entries, _), tape = entry_plan.entries(optimized_parameters)
    carrier = np.asarray(carrier)[:, 0]
    noise, noise_bar = np.asarray(noise).reshape(F), np.asarray(noise_bar, float).reshape(F)
    MU = np.asarray(MU)
    detector = dict(selection=entry_plan.selection(optimized_parameters), width=Z.shape[2])

    def general_adjoint(signal_bar):
        # A cotangent outside the detector rows: factorize the complete signal systems once more.
        plan = SystemPlan(A, n, noise_base=False, **detector)
        factors = plan.signal_factors(plan.signal_values(signal_entries, carrier))
        return plan.solve_transposed(factors, signal_bar)

    entries_bar = backward(
        SystemPlan(A, n, signal_base=False, noise_base=False, **detector),
        carrier_entries,
        signal_entries,
        carrier,
        np.asarray(signal).T,
        noise,
        np.asarray(Y),
        MU,
        np.asarray(Z),
        np.asarray(carrier_bar)[:, 0],
        np.asarray(signal_bar).T,
        noise_bar,
        general_adjoint,
    )
    selection_bar = None
    if entry_plan.qhd_parameter_indices.size:  # a homodyne phase may be a design variable
        selection_bar = selection_cotangent(carrier, noise, MU, noise_bar)
    gradient = entry_plan.pullback(
        tape, *entries_bar, value_count=optimized_parameters.shape[0], optimized_parameters=optimized_parameters, selection_bar=selection_bar
    )
    return gradient.astype(optimized_parameters.dtype)


def _forward_callback(optimized_parameters, host):
    """
    The simulation outputs, the residuals of the reverse pass (Y, MU, Z and the packed host arrays
    that are not static) and what the reverse callback needs besides: static arrays and layout.
    """
    n, m, F, noise_shape = _output_shapes(host)
    width = _detector_width(host)
    static, layout, traced = _split_host(host)
    packed_integers, packed_reals = _packed_sizes(layout)
    result_shapes = [
        jax.ShapeDtypeStruct((n, 1), jnp.complex128),
        jax.ShapeDtypeStruct((m, F), jnp.complex128),
        jax.ShapeDtypeStruct((F,), jnp.float64),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m, width), jnp.complex128),
    ]
    if packed_integers:
        result_shapes.append(jax.ShapeDtypeStruct((packed_integers,), jnp.int64))
    if packed_reals:
        result_shapes.append(jax.ShapeDtypeStruct((packed_reals,), jnp.float64))
    carrier, signal, noise, *residuals = jax.pure_callback(
        functools.partial(_host_forward, n, width, static, layout),
        tuple(result_shapes),
        optimized_parameters,
        *traced,
        vmap_method="sequential",
    )
    return (carrier, signal, noise.reshape(noise_shape)), tuple(residuals), (static, layout)


def _is_primal(x):
    return isinstance(x, jax.custom_derivatives.CustomVJPPrimal)


def _zeros_if_symbolic(value):
    if isinstance(value, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(value.shape, value.dtype)
    return value


@jax.custom_vjp
def _fast(optimized_parameters, host, others):
    return _forward_callback(optimized_parameters, host)[0]


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
    outputs, residuals, (static, layout) = _forward_callback(p, host)
    constant = _Constant((layout, tuple((key, _static_key(array)) for key, array in static.items())))
    return outputs, (p, outputs, constant, *residuals)


def _fast_bwd(residuals, cotangents):
    cotangents = tuple(_zeros_if_symbolic(c) for c in cotangents)
    if len(residuals) == 1:
        return residuals[0](cotangents)
    p, outputs, constant, *helpers = residuals
    layout, static = constant.value
    gradient = jax.pure_callback(
        functools.partial(_host_backward, {key: static_key.array for key, static_key in static}, layout),
        jax.ShapeDtypeStruct(p.shape, p.dtype),
        p,
        *outputs,
        *helpers,
        *cotangents,
        vmap_method="sequential",
    )
    return gradient, None, None


_fast.defvjp(_fast_fwd, _fast_bwd, symbolic_zeros=True)


def simulate_fast(arrays):
    """Fast `simulate` for argument dicts accepted by `supports`."""
    host = {key: arrays[key] for key in ARRAY_KEYS + MATRIX_KEYS}
    others = {key: value for key, value in arrays.items() if key not in host and key != "optimized_parameters"}
    return _fast(arrays["optimized_parameters"], host, others)
