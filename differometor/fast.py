"""
Fast host-side simulation path: sparse LU solves with SuperLU and a hand-written adjoint.

`differometor.simulate.simulate` dispatches here for setups with exactly one quantum noise
detector (a `qnoised` or a balanced homodyne detector `qhd`) and a single carrier operating
point. Parameter links are covered for the linking functions listed in `LINK_RULES` (`jnp.sqrt`,
which is what amplitude modulation uses). Everything else uses the original JAX implementation.
A balanced homodyne detector reads two ports, the second one with a phase factor from its
`phase` parameter (`EntryPlan.selection`); the phase may be a design variable.

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
  `scipy.sparse.linalg` costs more time than the whole simulation. It is loaded the first time
  a factorization asks for it: with the native kernel (`NATIVE_LU`, further down) both systems
  go through that kernel and SuperLU is never loaded.
- All frequencies share one sparsity pattern. The column order is computed once (COLAMD) and
  reused; per frequency there is one factorization and two solve calls.
- The quantum noise of the (single) detector is `Re(y^H N y)` with `y = S^-H w`; the
  same solve is reused as the adjoint helper for the gradient.
- Reverse mode is written by hand. Cotangents follow JAX's convention for complex
  values (bilinear pairing), so signal adjoints solve with `S^-T`, not `S^-H`.

The JAX integration (`simulate_fast`) is a `jax.custom_vjp` around two
`jax.pure_callback`s. The forward callback returns every residual the backward pass needs
as arrays that flow through JAX, so the backward callback can rebuild everything from them.
It only does so when it has to: a forward pass that is followed by a pullback keeps its
work on the host in a small cache (`FORWARD_CACHE`), and the pullback finds it through a key
that travels with the residuals.

The index arrays and base matrices from the build step are host constants
(`differometor.sparse.Constant`). They have no leaves as pytrees, so they are neither operands
of the callbacks nor residuals: the callbacks receive the constants themselves, and only the
design variables, the base parameters and the sweep values travel through JAX. Entries that
were replaced by JAX arrays travel through JAX as before.
"""

import importlib.machinery
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import differometor.components as components
from differometor.components import UNIT_VACUUM, H_PLANCK, F0
from differometor.sparse import base_entries

NOISE_SCALE = UNIT_VACUUM * H_PLANCK * F0 * 0.25


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


# Both compiled SciPy modules are loaded the first time they are asked for. With the native
# kernel a run never gets there: it factorizes the carrier system and the signal systems itself.
_UNLOADED = object()
_superlu = _UNLOADED


def _superlu_module():
    global _superlu
    if _superlu is _UNLOADED:
        _superlu = _load_superlu()
    return _superlu


def _load_sparsetools():
    """
    SciPy's compiled CSR kernels (`csr_matvecs`, `csc_matvecs`), loaded like `_load_superlu`
    without importing `scipy.sparse`. Returns None when the module cannot be found; the fast path
    then skips the pre-elimination in `_CondensedLU` and factorizes the full systems.
    """
    name = "scipy.sparse._sparsetools"
    if name in sys.modules:
        return sys.modules[name]
    try:
        root = importlib.util.find_spec("scipy").submodule_search_locations[0]
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = os.path.join(root, "sparse", "_sparsetools" + suffix)
            if os.path.exists(path):
                loader = importlib.machinery.ExtensionFileLoader(name, path)
                spec = importlib.util.spec_from_loader(name, loader, origin=path)
                module = importlib.util.module_from_spec(spec)
                loader.exec_module(module)
                if hasattr(module, "csr_matvecs") and hasattr(module, "csc_matvecs"):
                    return module
    except Exception:
        pass
    return None


# Only `_CondensedLU` uses these kernels (`_csr_kernels`). None: not available.
_sparsetools = _UNLOADED


def _csr_kernels():
    global _sparsetools
    if _sparsetools is _UNLOADED:
        _sparsetools = _load_sparsetools()
    return _sparsetools


def _splu(size, values, row_indices, column_pointers, column_order="COLAMD", small_supernodes=False):
    """
    LU factorization of a square CSC matrix, with the options `scipy.sparse.linalg.splu` passes.

    `column_order="NATURAL"` keeps the given column order (like `splu(..., permc_spec="NATURAL")`).

    `small_supernodes` sets SuperLU's `Relax` and `PanelSize` to 1. The systems here have about
    3.5 entries per row. With the default relaxed supernodes SuperLU pads the factors with explicit
    zeros (13120 stored entries instead of 8330 for the benchmark setup), which makes both the
    factorization and every solve slower. Pivoting is unchanged (partial pivoting).
    """
    options = dict(DiagPivotThresh=None, ColPerm=column_order, PanelSize=None, Relax=None)
    if column_order == "NATURAL":
        options["SymmetricMode"] = True
    if small_supernodes:
        options["PanelSize"] = 1
        options["Relax"] = 1
    return _superlu_module().gstrf(
        size,
        row_indices.size,
        np.ascontiguousarray(values, dtype=np.complex128),
        row_indices,
        column_pointers,
        csc_construct_func=None,
        ilu=False,
        options=options,
    )


# A signal cotangent counts as a multiple of the detector weights when the rest is below this share
# of it (see `_signal_adjoint`). Rounding leaves about 1e-16.
PARALLEL_TOLERANCE = 1e-13

# Rows per step where a product of two (rows, F) arrays is formed only to be stored or added
# somewhere else. The intermediate array of one step (200 kB for 101 frequencies) is reused by
# the next one instead of touching fresh memory for the whole product.
_ROW_BLOCK = 128


# ---------------------------------------------------------------------
# Component functions on dual numbers
# ---------------------------------------------------------------------


class Dual:
    """
    Value plus derivatives with respect to k real inputs; `tan` has shape (k, *val.shape) or is 0.

    Most operations in the component formulas combine a Dual with a plain number or array (a
    parameter that is no design variable, a constant). Those have no derivative, so the product
    rule needs one term only and a sum leaves the derivative as it is. Leaving the zero terms out
    also keeps a derivative that is the same at every frequency in its compact shape, when only
    the value picks up a frequency dependence. Duals are never changed in place, so a result may
    share its `tan` with an operand.
    """

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
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.tan + other.tan)
        return Dual(self.val + other, self.tan)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.tan - other.tan)
        return Dual(self.val - other, self.tan)

    def __rsub__(self, other):
        return Dual(other - self.val, -self.tan)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.tan * other.val + self.val * other.tan)
        return Dual(self.val * other, self.tan * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            quotient = self.val / other.val
            return Dual(quotient, (self.tan - quotient * other.tan) / other.val)
        return Dual(self.val / other, self.tan / other)

    def __rtruediv__(self, other):
        quotient = other / self.val
        return Dual(quotient, -(quotient * self.tan) / self.val)

    def __pow__(self, exponent):
        if isinstance(exponent, Dual):
            raise TypeError("Dual exponents are not supported")
        return Dual(self.val**exponent, exponent * self.val ** (exponent - 1) * self.tan)


def _chain(slope, tan):
    """`slope * tan` for a function whose slope can be infinite (sqrt at 0, arcsin at 1).

    At such a point (a mirror with reflectivity exactly 1 or 0) the derivative with respect to
    the function's own argument does not exist. The plain product would spoil every other
    direction too, because 0 * inf is nan: the tuning of that mirror would get a nan gradient.
    A direction in which the argument does not change (tangent exactly 0) keeps the derivative
    0, as in JAX's reverse mode, where such a direction never meets the infinite slope.
    """
    product = slope * tan
    if np.all(np.isfinite(slope)):
        return product
    return np.where(tan == 0, 0.0, product)


def _unary(function, derivative, singular=False, domain=None):
    def wrapped(x):
        if not isinstance(x, Dual):
            return function(x)
        y = function(x.val)
        return Dual(y, derivative(x.val, y) * x.tan)

    def wrapped_singular(x):
        if not isinstance(x, Dual):
            return function(_complex_outside(x, *domain))
        value = _complex_outside(x.val, *domain)
        y = function(value)
        return Dual(y, _chain(derivative(value, y), x.tan))

    return wrapped_singular if singular else wrapped


def _complex_outside(x, lower, upper):
    """A real `x` with a value outside [lower, upper] (no upper bound: None) as a complex array, anything else unchanged.

    The component functions get real numbers here, in the JAX path they get complex ones. For a
    real argument outside its domain NumPy's sqrt and arcsin return nan (a beamsplitter beyond the
    critical angle: n_left / n_right * sin(alpha) > 1, total internal reflection), JAX's complex
    functions return the value on the branch cut. With a zero imaginary part of either sign JAX
    takes the branch NumPy takes for +0, so `x + 0j` gives the same values and derivatives
    (`runs/philip/test_tir.py`).
    """
    if np.isrealobj(x) and (np.min(x) < lower or (upper is not None and np.max(x) > upper)):
        return np.asarray(x, dtype=complex)
    return x


class _NumpyShim:
    """Stand-in for `jax.numpy` inside the component functions (all inputs are real)."""

    pi, e, complex128, ndarray = np.pi, np.e, np.complex128, np.ndarray
    log10, log = staticmethod(np.log10), staticmethod(np.log)
    exp = staticmethod(_unary(np.exp, lambda x, y: y))
    sqrt = staticmethod(_unary(np.sqrt, lambda x, y: 0.5 / y, singular=True, domain=(0.0, None)))
    sin = staticmethod(_unary(np.sin, lambda x, y: np.cos(x)))
    cos = staticmethod(_unary(np.cos, lambda x, y: -np.sin(x)))
    sinh = staticmethod(_unary(np.sinh, lambda x, y: np.cosh(x)))
    cosh = staticmethod(_unary(np.cosh, lambda x, y: np.sinh(x)))
    arcsin = staticmethod(_unary(np.arcsin, lambda x, y: 1.0 / np.sqrt(1.0 - x * x), singular=True, domain=(-1.0, 1.0)))
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


def _group_sum(values, order, starts):
    """Sum the rows of `values` (E, R) that share a target slot (`order`/`starts` from `_grouping`)."""
    return np.add.reduceat(values[order], starts, axis=0)


def _grouping(targets):
    """Sort order, group starts and the distinct target slots for a scatter-add to `targets`."""
    order = np.argsort(targets, kind="stable")
    slots, starts = np.unique(targets[order], return_index=True)
    return order, starts, slots


class _EntryBlock:
    """One group of matrix entries (carrier, signal or noise) produced by component functions.

    `changing` lists the parameter columns that differ between the rows (the frequency sweep).
    Calls that read none of them give the same entries in every row and are evaluated once.

    `differentiable` marks the parameter slots that receive a design variable. The reverse pass
    only needs derivatives with respect to those, so every other function input is evaluated as a
    plain value (a mirror surface has 7 inputs, but only reflectivity and tuning are design
    variables; force terms have none).
    """

    def __init__(self, function_input_indices, function_indices, output_indices, changing=(), differentiable=None):
        self.inputs = np.asarray(function_input_indices)
        function_indices = np.asarray(function_indices)
        self.outputs = np.asarray(output_indices)
        if differentiable is None:
            needed = np.ones(self.inputs.shape, bool)
        else:
            needed = differentiable[self.inputs]
        swept = np.isin(self.inputs, np.asarray(changing, int))
        self.varying = swept.any(axis=1)
        self.groups = []  # (function index, members, varying, differentiated input columns)
        self.swept = []  # per group: which function inputs change along the rows
        for k in np.unique(function_indices):
            for varying in (True, False):
                members = np.nonzero((function_indices == k) & (self.varying == varying))[0]
                if members.size:
                    columns = np.nonzero(needed[members].any(axis=0))[0]
                    self.groups.append((int(k), members, varying, columns))
                    self.swept.append(swept[members].any(axis=0))
        # Per group and per function output that some entry reads: the entries (sorted by call), the
        # call each of them reads, the distinct calls and where each call starts among the entries
        # (several entries may read the same output of the same call).
        # `wanted`: the entries whose cotangent `back` reads (those of groups with a design variable
        # among their inputs), in the order it reads them; `slices` is the part of each group.
        # `entry_varying` marks the entries that differ between the rows.
        self.readers, wanted, self.slices, count = [], [], [], 0
        self.entry_varying = np.zeros(self.outputs.shape[1], bool)
        for _, members, varying, columns in self.groups:
            local = np.full(self.inputs.shape[0], -1)
            local[members] = np.arange(members.size)
            entries = np.nonzero(local[self.outputs[0]] >= 0)[0]
            call, output = local[self.outputs[0][entries]], self.outputs[1][entries]
            order = np.lexsort((call, output))
            entries, call, output = entries[order], call[order], output[order]
            parts = []
            for o in np.unique(output):
                pick = output == o
                calls, starts = np.unique(call[pick], return_index=True)
                parts.append((int(o), entries[pick], call[pick], calls, starts))
            self.readers.append(parts)
            self.entry_varying[entries] = varying
            read = entries.size if columns.size else 0
            self.slices.append((count, count + read))
            wanted.append(entries[:read])
            count += read
        self.wanted = np.concatenate(wanted) if wanted else np.empty(0, np.intp)

    # Entries are stored entry by entry, shape (E, R): one contiguous row of R values (the sweep)
    # per matrix entry. Selecting or writing a set of entries then moves whole rows.

    def _inputs(self, parameters, members, swept):
        """
        Function inputs of the calls `members` for parameters (R, P), and the shape of the result.

        One array per input: (members, R) for an input that changes along the rows (`swept`),
        (members, 1) for one that does not. The component functions are scalar formulas and
        broadcast, so every part of them that does not depend on the sweep is computed once per
        call instead of once per row. A group without swept inputs gives a single row.
        """
        indices = self.inputs[members]
        first, by_slot = parameters[0], parameters.T
        arguments = [by_slot[indices[:, j]] if swept[j] else first[indices[:, j], None] for j in range(indices.shape[1])]
        return arguments, (members.size, parameters.shape[0] if swept.any() else 1)

    def values(self, parameters, store=None):
        """parameters: (R, P) -> entries (E, R), no derivatives."""
        return self.run(parameters, derivatives=False, store=store)[0]

    def run(self, parameters, derivatives=True, store=None):
        """
        parameters: (R, P). Returns the entries (E, R) and the tape for the reverse pass.

        The tape has, per group and per output that is read, the derivative of that output with
        respect to the differentiated inputs, shape (inputs, calls, R) or (inputs, calls, 1) when
        it is the same in every row. A group without differentiated inputs has None, and so has an
        output that depends on none of them.

        With `store`, an array and the row of it that every entry belongs to (no row twice), the
        entries are written there instead of into an array of their own, and None is returned
        in their place.
        """
        if store is None:
            entries, rows = np.empty((self.outputs.shape[1], parameters.shape[0]), complex), None
        else:
            entries, rows = store
        tape = []
        for (function_index, members, _, columns), swept, parts in zip(self.groups, self.swept, self.readers):
            arguments, lead = self._inputs(parameters, members, swept)
            width = columns.size if derivatives else 0
            for position, j in enumerate(columns[:width]):  # only these inputs carry a derivative
                seed = np.zeros((width, 1, 1))
                seed[position] = 1.0
                arguments[j] = Dual(arguments[j], seed)
            outputs = FUNCTIONS_NUMPY[function_index](arguments)
            slopes = []
            for o, readers, call, _, _ in parts:
                item = outputs[o]
                value, slope = (item.val, item.tan) if isinstance(item, Dual) else (item, 0.0)
                entries[readers if rows is None else rows[readers]] = np.broadcast_to(value, lead)[call]
                if np.ndim(slope) == 0:  # a constant: no derivative
                    slopes.append(None)
                else:
                    slopes.append(np.broadcast_to(slope, (width, lead[0], slope.shape[-1])))
            tape.append(slopes if width else None)
        return (entries if rows is None else None), tape

    def back(self, tape, entries_bar, parameter_count, compact=False):
        """entries_bar: (E, R) -> real parameter cotangent summed over the rows (P,).

        With `compact`, `entries_bar` has one row per entry of `self.wanted`, in that order.
        """
        parameters_bar = np.zeros(parameter_count)
        for (function_index, members, varying, columns), slopes, parts, (start, _) in zip(
            self.groups, tape, self.readers, self.slices
        ):
            if slopes is None:  # no design variable among the inputs of this group
                continue
            inputs_bar = np.zeros((columns.size, members.size))
            for (o, readers, _, calls, starts), slope in zip(parts, slopes):
                stop = start + readers.size
                if slope is not None:
                    bar = entries_bar[start:stop] if compact else entries_bar[readers]
                    if starts.size != readers.size:  # entries that read the same output of the same call
                        bar = np.add.reduceat(bar, starts, axis=0)
                    if calls.size != members.size:
                        slope = slope[:, calls]
                    if slope.shape[2] == 1:  # the same derivative in every row
                        inputs_bar[:, calls] += (bar.sum(axis=1) * slope[:, :, 0]).real
                    else:
                        inputs_bar[:, calls] += np.einsum("cr,jcr->jc", bar, slope).real
                start = stop
            parameters_bar += np.bincount(self.inputs[members][:, columns].T.ravel(), inputs_bar.ravel(), parameter_count)
        return parameters_bar


# Linking functions (derived parameters, e.g. a signal amplitude that follows `sqrt(laser power)`)
# that the fast path evaluates on the host: the NumPy value and the derivative as a function of
# input and value. Setups with any other linking function use the original JAX implementation.
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
        # Balanced homodyne detector (qhd): the entries `qhd_places` of the noise detector's
        # selection vector carry the factor exp(1j * phase), phase in degrees (`selection`).
        self.qhd_parameter_indices = np.asarray(A["qhd_parameter_indices"], np.intp).reshape(-1)
        self.qhd_places = np.asarray(A["qhd_placing_indices"], np.intp).reshape(-1, 3)[:, 2]
        self.selection_base = np.asarray(A["noise_selection_vectors"])[0, 0]
        # Only a phase that is a design variable needs the cotangent of the selection vector.
        self.phase_is_design_variable = bool(np.isin(self.qhd_parameter_indices, self.optimized_parameter_indices).any())
        # Parameter slots whose derivative reaches a design variable (see `pullback`).
        differentiable = np.zeros(self.base.size, bool)
        differentiable[self.optimized_parameter_indices] = True
        differentiable[self.link_targets] = differentiable[self.link_sources]
        # A phase that follows a design variable through a link needs that cotangent as well.
        if self.link_targets.size and np.isin(self.qhd_parameter_indices, self.link_targets[differentiable[self.link_targets]]).any():
            self.phase_is_design_variable = True
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
        The selection vector of the noise detector, shape (m,). A balanced homodyne detector
        multiplies the entries of its second port by exp(1j * phase), as `simulate_jax` does
        with `lax.scatter_mul`. Without one this is the stored vector itself.
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

    def values(self, optimized_parameters, signal_store=None):
        """Carrier, signal and noise entries without derivatives (forward pass).

        `signal_store`: where the signal entries go instead of an array of their own (see `_EntryBlock.run`).
        """
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        return self.carrier.values(carrier_rows)[:, 0], self.signal.values(signal_rows, signal_store), self.noise.values(signal_rows)

    def entries(self, optimized_parameters, signal_store=None):
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        carrier, carrier_tape = self.carrier.run(carrier_rows)
        signal, signal_tape = self.signal.run(signal_rows, store=signal_store)
        noise, noise_tape = self.noise.run(signal_rows)
        return (carrier[:, 0], signal, noise), (carrier_tape, signal_tape, noise_tape)

    def signal_into(self, optimized_parameters, signal_store):
        """Only the signal entries, without derivatives, written into `signal_store`."""
        self.signal.values(self._rows(optimized_parameters)[1], signal_store)

    def pullback(self, tape, carrier_bar, signal_bar, noise_bar, value_count, compact=False, optimized_parameters=None, selection_bar=None):
        """Gradient with respect to the design variables from the cotangents of the entries.

        With `compact`, `signal_bar` only has the rows of `self.signal.wanted` (see `backward`).
        `selection_bar` is the cotangent of `selection(optimized_parameters)`; it is needed when
        the phase of a balanced homodyne detector is a design variable.
        """
        carrier_tape, signal_tape, noise_tape = tape
        parameter_count = self.base.size
        signal_rows_bar = self.signal.back(signal_tape, signal_bar, parameter_count, compact) + self.noise.back(
            noise_tape, noise_bar, parameter_count
        )
        signal_rows_bar[self.signal_changing_indices] = 0.0  # overwritten by the sweep values
        P_bar = self.carrier.back(carrier_tape, carrier_bar[:, None], parameter_count) + signal_rows_bar
        P_bar[self.carrier_changing_indices] = 0.0
        if selection_bar is not None and self.qhd_parameter_indices.size:
            # selection = base * exp(1j * radians(phase)), so d selection / d phase = 1j * pi / 180 * selection.
            places = self.qhd_places
            phase_bar = (selection_bar[places] * 1j * self.selection(optimized_parameters)[places]).real * (np.pi / 180)
            np.add.at(P_bar, self.qhd_parameter_indices, phase_bar)
        if self.link_targets.size:
            # target = rule(source): the cotangent of a linked slot goes to its source.
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


class _StoredEntries:
    """
    Signal entries that were written straight into the value store of the signal matrices
    (`SystemPlan.signal_store`) instead of into an array of their own, which would only be copied
    there. `take()` hands the store over. The factorization works in it afterwards, so whoever
    asks again gets a new store from `fill`.
    """

    def __init__(self, store, fill):
        self._store, self._fill = store, fill

    def take(self):
        store, self._store = self._store, None
        return self._fill() if store is None else store


class SystemPlan:
    """Static sparsity data derived from the index arrays (cheap, rebuilt in every callback)."""

    def __init__(self, A):
        get = lambda key: np.asarray(A[key])
        # Base matrices arrive as sparse entries (`BaseMatrix`) or, from older callers, dense.
        carrier_index, carrier_values, (n, _) = base_entries(A["carrier_matrix"])
        signal_index, signal_values, (m, _) = base_entries(A["signal_matrix"])
        self.n, self.m = n, m
        # Carrier system: pattern of the n x n matrix in CSC order (key `column * n + row`),
        # right-hand side in column n. The pattern has the nonzero base entries and every entry
        # a component function writes; no dense n x (n + 1) array is built for it.
        self.cmi = get("carrier_matrix_indices") % np.array([[n], [n + 1]])
        self.carrier_live = _last_writer(self.cmi[0], self.cmi[1], n + 1)
        base_rows, base_cols = np.divmod(carrier_index, n + 1)
        in_base = (base_cols < n) & (carrier_values != 0)
        in_matrix = self.cmi[1] < n
        base_keys = base_cols[in_base] * n + base_rows[in_base]
        written_keys = self.cmi[1][in_matrix] * n + self.cmi[0][in_matrix]
        keys = np.unique(np.concatenate([base_keys, written_keys]))
        self.carrier_cols, rows = np.divmod(keys, n)
        self.carrier_rows = rows.astype(np.intc)
        self.carrier_ptr = np.searchsorted(self.carrier_cols, np.arange(n + 1)).astype(np.intc)
        self.carrier_base_values = np.zeros(keys.size, complex)
        self.carrier_base_values[np.searchsorted(keys, base_keys)] = carrier_values[in_base]
        self.carrier_base_rhs = np.zeros(n, complex)
        self.carrier_base_rhs[base_rows[base_cols == n]] = carrier_values[base_cols == n]
        # Where the entries that survive (last write wins) go: a pattern slot or a row of the right-hand side.
        to_matrix, to_rhs = self.carrier_live & in_matrix, self.carrier_live & ~in_matrix
        self.carrier_matrix_entries = np.nonzero(to_matrix)[0]
        self.carrier_matrix_slots = np.searchsorted(keys, self.cmi[1][to_matrix] * n + self.cmi[0][to_matrix])
        self.carrier_rhs_entries = np.nonzero(to_rhs)[0]
        self.carrier_rhs_rows = self.cmi[0][to_rhs]

        # Signal system: pattern of the augmented m x (m + 1) matrix in CSC order. Entries are
        # identified by the key `column * m + row`, which sorts them column by column.
        smi = get("signal_matrix_indices") % np.array([[m], [m + 1]])
        placing = get("signal_carrier_placing_indices") % np.array([[m], [m + 1]])
        self.signal_carrier_indices = get("signal_carrier_indices").ravel()
        key = lambda rows, cols: cols * m + rows
        base_keys = key(*np.divmod(signal_index, m + 1))
        keys = np.unique(np.concatenate([base_keys, key(smi[0], smi[1]), key(placing[0], placing[1])]))
        cols, rows = np.divmod(keys, m)
        self.rows, self.cols, self.size = rows, cols, rows.size
        self.signal_base = np.zeros(keys.size, complex)
        self.signal_base[np.searchsorted(keys, base_keys)] = signal_values
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
        noise_index, noise_values, _ = base_entries(A["noise_matrix"])
        nmi = get("noise_system_matrix_indices") % m
        self.noise_live = _last_writer(nmi[0], nmi[1], m)
        constant = ~np.isin(noise_index, nmi[0] * m + nmi[1])
        base_rows, base_cols = np.divmod(noise_index[constant], m)
        self.noise_rows = np.concatenate([base_rows, nmi[0][self.noise_live]])
        self.noise_cols = np.concatenate([base_cols, nmi[1][self.noise_live]])
        self.noise_base = noise_values[constant]
        self.noise_by_row = _grouping(self.noise_rows)
        self.noise_by_col = _grouping(self.noise_cols)
        self.selection = get("noise_selection_vectors")[0, 0]
        self._layout = None

    def carrier_matrix(self, carrier_entries):
        """Factorization of the carrier matrix and its right-hand side (n,).

        The factorization has a `solve(b, trans="N")` for b of shape (n,). With `NATIVE_LU` it
        comes from the native kernel (`_SingleLU`), otherwise or when the kernel declines from SuperLU.
        """
        values = self.carrier_base_values.copy()
        values[self.carrier_matrix_slots] = carrier_entries[self.carrier_matrix_entries]
        rhs = self.carrier_base_rhs.copy()
        rhs[self.carrier_rhs_rows] = carrier_entries[self.carrier_rhs_entries]
        if NATIVE_LU:
            factors, _ = _PatternLU.create(self.n, self.carrier_rows, self.carrier_cols, values[:, None])
            if factors is not None:
                return _SingleLU(factors), rhs
        return _splu(self.n, values, self.carrier_rows, self.carrier_ptr), rhs

    @property
    def spare(self):
        """Rows behind the signal values that let the factorization work in the same array."""
        return max(0, _CondensedLU.room(self.matrix_rows.size) - self.size)

    def signal_store(self, F, spare=0):
        """The array for the values of the augmented signal matrices, shape (size, F), with the constant entries in place."""
        V = np.empty((self.size + spare, F), complex)[: self.size]
        V[:] = self.signal_base[:, None]
        return V

    def signal_values(self, signal_entries, carrier, spare=0):
        """
        Nonzero values of the augmented signal matrices, shape (size, F), plus placing pre-values.

        With `spare`, the values are the first rows of a larger array (`V.base`) with that many
        more rows, so that `_CondensedLU` can work in it without copying them.
        """
        if isinstance(signal_entries, _StoredEntries):  # already in place, see `_run_forward`
            V = signal_entries.take()
        else:
            V = self.signal_store(signal_entries.shape[1], spare)
            if self.update_live.all():
                V[self.update_positions] = signal_entries
            else:
                V[self.update_positions[self.update_live]] = signal_entries[self.update_live]
        pre = V[self.placing_positions]
        # Placing entries become -pre * carrier. The sign goes on the short carrier factor and the
        # rows are written a block at a time, so no further arrays of the size of `pre` appear.
        live = np.nonzero(self.placing_live)[0]
        positions = self.placing_positions[live]
        factors = -carrier[self.signal_carrier_indices[live]]
        for start in range(0, live.size, _ROW_BLOCK):
            block = slice(start, start + _ROW_BLOCK)
            V[positions[block]] = pre[live[block]] * factors[block, None]
        np.negative(V.imag, out=V.imag, where=self.conjugated[:, None])  # conjugate those rows in place
        return V, pre

    def signal_factors(self, V, overwrite=False, restore=None):
        """Solver for S_f x = b and S_f^T z = b at every frequency (rows of `V`), see `_CondensedLU`.

        With `NATIVE_LU` the native kernel is tried first (`_PatternLU`). It works in place like
        `_CondensedLU`, so after a failed attempt with `overwrite` the values are gone: `restore`
        must then return them again (same layout as `V`).
        """
        if NATIVE_LU:
            # First with the pivots chosen from a few frequencies; if a pivot then fails the check
            # at another frequency, once more with the choice made on all of them.
            for samples in (None, 0):
                factors, status = _PatternLU.create(self.m, self.matrix_rows, self.matrix_cols, V, self.in_matrix, overwrite, samples)
                if factors is not None:
                    return factors
                if overwrite and status != -1:
                    if restore is None:
                        raise ValueError("signal_factors(overwrite=True) needs `restore` when NATIVE_LU is on")
                    V = restore()
                if status != _PatternLU.UNSTABLE:
                    break
        return _CondensedLU(self.m, self.matrix_rows, self.matrix_cols, V, self.in_matrix, overwrite)

    def reverse_layout(self, wanted=None, varying=None):
        """The `_ReverseLayout` for these signal entries (kept for the next call with the same ones)."""
        if self._layout is None or self._layout[0] is not wanted or self._layout[1] is not varying:
            self._layout = (wanted, varying, _ReverseLayout(self, wanted, varying))
        return self._layout[2]

    def noise_values(self, noise_entries):
        """Noise matrix values, shape (entries, F), from the noise entries (E, F)."""
        F = noise_entries.shape[1]
        base = np.broadcast_to(self.noise_base.astype(complex)[:, None], (self.noise_base.size, F))
        return np.concatenate([base, noise_entries[self.noise_live]])

    def detector_weights(self, carrier):
        n, s = self.n, self.selection
        w = s.astype(complex).copy()
        w[:n] = s[:n] * carrier
        w[n : 2 * n] = np.conj(s[n : 2 * n] * carrier)
        return w


class _BlockLU:
    """
    LU factorizations of F square matrices that share one sparsity pattern.

    The matrices are factorized as one block-diagonal system diag(B_0, ..., B_{F-1}) per chunk of
    frequencies, with a single SuperLU call instead of F. Pivoting never leaves a block, so the
    factors and every solution are bit-identical to F separate factorizations. What is saved is
    the fixed cost of each SuperLU call (setup, work arrays, Python round trip).
    """

    MAX_ROWS = 1 << 18  # rows per SuperLU call; bounds the size of one block-diagonal system

    def __init__(self, m, values, rows, pointers):
        F, entries = values.shape
        self.m, self.F = m, F
        step = max(1, self.MAX_ROWS // m)
        self.chunks = []
        for start in range(0, F, step):
            count = min(step, F - start)
            blocks = np.arange(count)[:, None]
            block_rows = (rows[None, :] + m * blocks).astype(np.intc).ravel()
            block_ptr = np.empty(count * m + 1, np.intc)
            block_ptr[:-1] = (pointers[None, :-1] + entries * blocks).ravel()
            block_ptr[-1] = count * entries
            block_values = values[start : start + count].ravel()
            lu = _splu(count * m, block_values, block_rows, block_ptr, "NATURAL", small_supernodes=True)
            self.chunks.append((start, count, lu))

    def solve(self, b, trans="N"):
        """Solve B_f x_f = b_f (or the transposed systems) for b of shape (F, m, k)."""
        m, k = self.m, b.shape[2]
        x = np.empty((self.F, m, k), complex)
        for start, count, lu in self.chunks:
            block = np.ascontiguousarray(b[start : start + count], dtype=complex).reshape(count * m, k)
            x[start : start + count] = lu.solve(block, trans=trans).reshape(count, m, k)
        return x


def _largest_square(values):
    """max over the frequencies of |v|^2 for entry values of shape (entries, F)."""
    parts = np.ascontiguousarray(values).view(np.float64).reshape(values.shape[0], values.shape[1], 2)
    return np.einsum("efk,efk->ef", parts, parts).max(axis=1, initial=0.0)


def _eliminable(size, rows, cols, square, one):
    """
    Mask of unknowns that can be eliminated exactly before the LU, without pivoting.

    Unknown i qualifies when the diagonal entry of row i is exactly 1 at every frequency and
    either all other entries of row i have modulus <= 1, or no other row refers to unknown i.
    Subtracting multiples of such a row from another row never enlarges that row beyond its own
    scale, so the elimination is as stable as LU with pivoting. The selected unknowns are not
    coupled to each other: their block of the matrix is the identity.

    `square` is the largest |v|^2 of every entry over the frequencies, `one` marks entries that
    are exactly 1 everywhere. Also returns the mask of entries that are nonzero somewhere.
    """
    diagonal = rows == cols
    unit = np.zeros(size, bool)
    unit[rows[diagonal]] = one[diagonal]
    live = square > 0
    off = live & ~diagonal
    r, c = rows[off], cols[off]
    row_max = np.zeros(size)
    np.maximum.at(row_max, r, square[off])
    row_count, column_count = np.bincount(r, minlength=size), np.bincount(c, minlength=size)
    undecided = unit & ((row_max <= 1 + 1e-9) | (column_count == 0))
    # Independent set, built in rounds: an undecided unknown is picked when it has the lowest
    # priority (expected fill, then index) among its undecided neighbours.
    priority = (row_count * column_count).astype(np.int64) * size + np.arange(size)
    last = np.iinfo(np.int64).max
    picked = np.zeros(size, bool)
    while undecided.any():
        p = np.where(undecided, priority, last)
        lowest = p.copy()
        np.minimum.at(lowest, r, p[c])
        np.minimum.at(lowest, c, p[r])
        new = undecided & (lowest == p)
        picked |= new
        neighbours = np.zeros(size, bool)
        neighbours[r[new[c]]] = True
        neighbours[c[new[r]]] = True
        undecided &= ~(new | neighbours)
    return picked, live


class _BlockCSR:
    """
    The sparse matrices -A_f of all frequencies as one CSR matrix over the index (f, unknown).

    `subtract(x)` computes x[rows] -= A x[cols] in place for x of shape (F, size, k), and with
    `transposed=True` x[cols] -= A^T x[rows]. Rows and columns of A must not overlap.
    """

    def __init__(self, size, rows, cols, values):
        entries, F = values.shape
        order = np.argsort(rows, kind="stable")
        count = np.bincount(rows, minlength=size)
        blocks = np.arange(F)[:, None]
        self.pointers = np.empty(F * size + 1, np.intc)
        self.pointers[:-1] = ((np.cumsum(count) - count)[None, :] + entries * blocks).ravel()
        self.pointers[-1] = entries * F
        self.indices = (cols[order][None, :] + size * blocks).astype(np.intc).ravel()
        self.data = np.ascontiguousarray(-values[order].T).ravel()
        self.total = F * size

    def subtract(self, x, transposed=False):
        flat = x.reshape(-1)
        kernels = _csr_kernels()
        kernel = kernels.csc_matvecs if transposed else kernels.csr_matvecs
        kernel(self.total, self.total, x.shape[2], self.pointers, self.indices, self.data, flat, flat)


class _Elimination:
    """
    One exact elimination step. With the unknowns ordered as (I, J) the matrix is

        S = [[1, S_IJ], [S_JI, S_JJ]],

    so x_I = b_I - S_IJ x_J and the remaining system is (S_JJ - S_JI S_IJ) x_J = b_J - S_JI b_I.
    The transposed system reduces to the transpose of the same matrix.

    Unknowns keep their numbers. The entry values of all stages live in one array `store` of
    shape (capacity, F); `ids` are the rows of `store` for the pattern (`rows`, `cols`), which is
    in CSC order. Entries of S_JJ that the step changes are updated in place, fill-in is appended.
    """

    def __init__(self, size, rows, cols, ids, picked, live, store, used):
        row_in, col_in = picked[rows], picked[cols]
        ij = np.nonzero(live & row_in & ~col_in)[0]
        ji = np.nonzero(live & ~row_in & col_in)[0]
        jj = np.nonzero(live & ~row_in & ~col_in)[0]
        self.upper = _BlockCSR(size, rows[ij], cols[ij], store[ids[ij]])  # S_IJ
        self.lower = _BlockCSR(size, rows[ji], cols[ji], store[ids[ji]])  # S_JI
        # S_JI S_IJ: every entry (r, i) meets every entry (i, j) of the same eliminated unknown i.
        by_unknown = np.argsort(rows[ij], kind="stable")
        count = np.bincount(rows[ij], minlength=size)
        start = np.cumsum(count) - count
        unknown = cols[ji]
        partners = count[unknown]
        left = np.repeat(np.arange(ji.size), partners)
        offset = np.arange(left.size) - np.repeat(np.cumsum(partners) - partners, partners)
        right = by_unknown[start[unknown[left]] + offset]
        keys, self.ids, self.changed, self.used = cols[jj] * size + rows[jj], ids[jj], ids[:0], used
        if left.size:
            order, starts, targets = _grouping(cols[ij[right]] * size + rows[ji[left]])
            at = np.minimum(np.searchsorted(keys, targets), max(keys.size - 1, 0))
            exists = keys[at] == targets if keys.size else np.zeros(targets.size, bool)
            fresh = used + np.arange(targets.size - exists.sum())
            self.used = used + fresh.size
            self.fits = self.used <= store.shape[0]
            if not self.fits:
                return
            sums = np.add.reduceat(store[ids[ji[left[order]]]] * store[ids[ij[right[order]]]], starts, axis=0)
            touched = self.ids[at[exists]]
            store[touched] -= sums[exists]
            store[fresh] = 0.0
            store[fresh] -= sums[~exists]
            self.changed = np.concatenate([touched, fresh])
            keys, self.ids = np.concatenate([keys, targets[~exists]]), np.concatenate([self.ids, fresh])
            order = np.argsort(keys, kind="stable")
            keys, self.ids = keys[order], self.ids[order]
        self.fits = True
        self.cols, self.rows = np.divmod(keys, size)


class _CondensedLU:
    """
    Solves S_f x = b and S_f^T z = b for F matrices that share one sparsity pattern.

    SuperLU spends a fixed amount of work on every column, and for systems this sparse (about
    3.5 entries per row) that fixed work dominates. Most unknowns of an interferometer signal
    matrix are simple, though: sources, read-only ports and fields that are a combination of a
    few others with coefficients of modulus <= 1. `_eliminable` picks such unknowns, and
    `_Elimination` removes them exactly with array operations over all frequencies. This repeats
    while a stage still removes a good share of the system. SuperLU (with partial pivoting, as
    before) only factorizes what is left: 200 of 751 unknowns for the benchmark setup.

    `values` has shape (entries, F) for the CSC pattern (`rows`, `cols`). With `select`, a
    boolean mask, `values` has one row per mask element and only the selected rows are entries.
    Right-hand sides and solutions have shape (F, size, k).
    """

    MAX_STAGES = 3
    MIN_SHARE = 0.25  # stop when a stage would remove less than this share of the unknowns
    ORDERING = "MMD_ATA"  # 5-10 % less SuperLU work than COLAMD on the reduced systems

    @staticmethod
    def room(entries):
        """Rows of the value store for a pattern with this many entries: all stages, with room for fill-in."""
        return 3 * entries + 256

    def __init__(self, size, rows, cols, values, select=None, overwrite=False):
        entries, F = rows.size, values.shape[1]
        self.size, self.F = size, F
        # With `overwrite` the caller gives the values up. If they are the head of a larger array
        # with enough room behind them (`SystemPlan.signal_values(..., spare=...)`) and the entries
        # are its first rows, the stages work in that array. Otherwise the values are copied.
        store = values.base if overwrite else None
        in_place = (
            isinstance(store, np.ndarray)
            and store.dtype == np.complex128
            and store.shape == (max(store.shape[0], self.room(entries)), F)
            and store.flags.c_contiguous
            and store.__array_interface__["data"][0] == values.__array_interface__["data"][0]
            and (select is None or bool(select[:entries].all()))
        )
        if not in_place:
            store = np.empty((self.room(entries), F), complex)
            if select is None:
                store[:entries] = values
            else:
                np.compress(select, values, axis=0, out=store[:entries])
        ids, used = np.arange(entries), entries
        self.stages = []
        remaining = np.ones(size, bool)
        if _csr_kernels() is not None and self.MAX_STAGES:
            square = np.zeros(store.shape[0])
            one = np.zeros(store.shape[0], bool)
            square[:entries] = _largest_square(store[:entries])
            diagonal = np.nonzero(rows == cols)[0]
            one[diagonal] = (store[diagonal] == 1).all(axis=1)
            while len(self.stages) < self.MAX_STAGES:
                picked, live = _eliminable(size, rows, cols, square[ids], one[ids])
                if picked.sum() < self.MIN_SHARE * remaining.sum():
                    break
                stage = _Elimination(size, rows, cols, ids, picked, live, store, used)
                if not stage.fits:
                    break
                self.stages.append(stage)
                remaining &= ~picked
                rows, cols, ids, used = stage.rows, stage.cols, stage.ids, stage.used
                changed = store[stage.changed]
                square[stage.changed] = _largest_square(changed)
                one[stage.changed] = (changed == 1).all(axis=1)
        # What is left goes to SuperLU, renumbered 0..n-1. The column order comes from one
        # matrix and is reused for all frequencies (shared pattern).
        self.rest = rest = np.nonzero(remaining)[0]
        n = rest.size
        number = np.cumsum(remaining) - 1
        rows, cols = number[rows].astype(np.intc), number[cols]
        pointers = np.searchsorted(cols, np.arange(n + 1)).astype(np.intc)
        first = _splu(n, np.ascontiguousarray(store[ids, 0]), rows, pointers, self.ORDERING)
        order = np.argsort(first.perm_c)
        counts = np.diff(pointers)[order]
        ordered_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.intc)
        gather = np.repeat(pointers[order] - ordered_ptr[:-1], counts) + np.arange(ordered_ptr[-1])
        self.lu = _BlockLU(n, np.ascontiguousarray(store[ids[gather]].T), rows[gather], ordered_ptr)
        self.ordered = rest[order]

    def solve(self, b, trans="N", overwrite=False):
        """x with S_f x_f = b_f (`trans="N"`) or S_f^T x_f = b_f (`trans="T"`).

        With `overwrite`, a contiguous complex `b` is used as the work array and returned.
        """
        transposed = trans == "T"
        if overwrite and isinstance(b, np.ndarray) and b.dtype == np.complex128 and b.flags.c_contiguous and b.flags.writeable:
            x = b
        else:
            x = np.array(b, dtype=complex, order="C")
        for stage in self.stages:  # right-hand side of the remaining system
            (stage.upper if transposed else stage.lower).subtract(x, transposed)
        if transposed:
            x[:, self.rest] = self.lu.solve(x[:, self.ordered], trans="T")
        else:
            x[:, self.ordered] = self.lu.solve(x[:, self.rest])
        for stage in reversed(self.stages):  # eliminated unknowns from the ones solved for
            (stage.lower if transposed else stage.upper).subtract(x, transposed)
        return x

    def solve_rows(self, b, trans="N"):
        """The same for one right-hand side of shape (size, F), one row per unknown; returns a new array."""
        x = self.solve(np.ascontiguousarray(b.T, dtype=complex)[:, :, None], trans, overwrite=True)
        return np.ascontiguousarray(x[:, :, 0].T)

    def solve_scattered(self, sides, trans="N"):
        """
        Solutions of shape (size, F), one row per unknown, for right-hand sides that are zero
        outside a few rows. `sides` has one list per right-hand side, with pairs of row numbers
        (no repeats within a pair) and values of shape (rows, F) or (rows, 1); the pairs of a
        right-hand side add up. All right-hand sides are solved in one pass.
        """
        x = np.zeros((self.F, self.size, len(sides)), complex)
        for j, parts in enumerate(sides):
            for rows, values in parts:
                x[:, rows, j] += values.T
        x = self.solve(x, trans, overwrite=True)
        return [np.ascontiguousarray(x[:, :, j].T) for j in range(len(sides))]


# ---------------------------------------------------------------------
# Native factorization with one pivot sequence for all frequencies (optional)
# ---------------------------------------------------------------------

# With `NATIVE_LU` the signal systems are factorized and solved by the C kernel in
# `_native/pattern_lu.c` (`_PatternLU`) instead of `_CondensedLU`. The switch is read in every
# host callback, so it can be changed at any time. It is on by default. The results
# differ from the SuperLU path in the last digits (about 1e-9 relative; both paths are that close
# to the exact solution). DIFFEROMETOR_NATIVE_LU=0 in the environment switches it off without a
# code change. When the shared library is missing or cannot be loaded, or when the kernel finds
# no pivot sequence that is stable at every frequency, `_CondensedLU` does the work as before.
NATIVE_LU = os.environ.get("DIFFEROMETOR_NATIVE_LU", "1") != "0"

_pattern_lu_library = None  # None: not tried yet, False: unavailable


def _pattern_lu():
    """The native kernel as a ctypes library, or None when it cannot be loaded."""
    global _pattern_lu_library
    if _pattern_lu_library is None:
        try:
            import ctypes

            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_native", "libpattern_lu.so")
            library = ctypes.PyDLL(path)  # keeps the GIL: a factorization is never used by two threads at once
            pointer, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
            library.plu_factor.argtypes = [ctypes.POINTER(pointer)] + [integer] * 3 + [pointer] * 3 + [integer] * 2 + [real] * 2 + [integer] * 2 + [pointer]
            library.plu_factor.restype = integer
            library.plu_free.argtypes, library.plu_free.restype = [pointer], None
            library.plu_solve.argtypes, library.plu_solve.restype = [pointer, pointer, pointer, integer], None
            library.plu_solve_fmk.argtypes, library.plu_solve_fmk.restype = [pointer, pointer, pointer, integer, integer], integer
            library.plu_stats.argtypes, library.plu_stats.restype = [pointer, pointer], None
            library.plu_select.argtypes, library.plu_select.restype = [integer], integer
            library.handle, library.byref = pointer, ctypes.byref
            _pattern_lu_library = library
        except Exception:
            _pattern_lu_library = False
    return _pattern_lu_library or None


class _PatternLU:
    """
    Solves S_f x = b and S_f^T z = b for F matrices that share one sparsity pattern, like
    `_CondensedLU`, with the native kernel `_native/pattern_lu.c`.

    All frequencies are factorized with one pivot sequence. The values of a matrix entry at all
    frequencies are one contiguous row, so every elimination step is a plain loop over that row:
    no index work, no pivot search and no symbolic analysis per frequency (SuperLU repeats all
    three for every frequency block). The sequence is chosen from `SAMPLES` frequencies: lowest
    Markowitz count among the entries that are at least `U_SELECT` times the largest entry of
    their row and of their column. The numeric factorization then checks every pivot against
    `U_VERIFY` at every frequency, so multipliers are bounded by 1 / U_VERIFY in both factors
    everywhere. Unit diagonal entries with small neighbours, the ones `_CondensedLU` eliminates
    beforehand, come first here as well, because they cost nothing.

    `create` returns None when the library is unavailable or no shared pivot sequence passes the
    checks; the caller then uses `_CondensedLU`. Same arguments and layouts as `_CondensedLU`.
    """

    U_SELECT = 0.1
    U_VERIFY = 0.025
    SAMPLES = 8
    UNSTABLE = 4  # status of the kernel: a pivot failed the check at a frequency outside the sample

    def __init__(self, library, handle, store, size, frequencies):
        self._library, self._handle, self._store = library, handle, store
        self.size, self.F = size, frequencies
        self._work = None  # right-hand side of `solve_scattered`, reused from call to call

    @classmethod
    def create(cls, size, rows, cols, values, select=None, overwrite=False, samples=None):
        """The factorization, or None with the status of the kernel (-1: no library)."""
        library = _pattern_lu()
        if library is None:
            return None, -1
        entries, F = rows.size, values.shape[1]
        # Like `_CondensedLU`: work in the caller's array when it is given up and has room behind
        # the entries, otherwise in a copy.
        room = _CondensedLU.room(entries)
        store = values.base if overwrite else None
        in_place = (
            isinstance(store, np.ndarray)
            and store.dtype == np.complex128
            and store.shape == (max(store.shape[0], room), F)
            and store.flags.c_contiguous
            and store.flags.writeable
            and store.__array_interface__["data"][0] == values.__array_interface__["data"][0]
            and (select is None or bool(select[:entries].all()))
        )
        if not in_place:
            store = np.empty((room, F), complex)
            if select is None:
                store[:entries] = values
            else:
                np.compress(select, values, axis=0, out=store[:entries])
        rows, cols = np.ascontiguousarray(rows, np.intc), np.ascontiguousarray(cols, np.intc)
        handle, info = library.handle(), np.zeros(2, np.intc)
        status = library.plu_factor(
            library.byref(handle),
            size,
            F,
            entries,
            rows.ctypes.data,
            cols.ctypes.data,
            store.ctypes.data,
            store.shape[0],
            entries,
            cls.U_SELECT,
            cls.U_VERIFY,
            cls.SAMPLES if samples is None else samples,
            0,
            info.ctypes.data,
        )
        if status:
            return None, status
        return cls(library, handle, store, size, F), 0

    def __del__(self):
        try:
            if self._handle:
                self._library.plu_free(self._handle)
                self._handle = None
        except Exception:  # interpreter shutdown
            pass

    def stats(self):
        """Counters of the kernel (entries of L and U, update operations, times in ns, ...)."""
        names = ("L", "U", "rows_used", "updates", "evaluations", "searched", "simd", "chunk", "unit_pivots", "ns_analysis", "ns_numeric")
        out = np.zeros(len(names), np.int64)
        self._library.plu_stats(self._handle, out.ctypes.data)
        return dict(zip(names, out.tolist()))

    def solve(self, b, trans="N", overwrite=False):
        """x with S_f x_f = b_f (`trans="N"`) or S_f^T x_f = b_f (`trans="T"`), shapes (F, size, k).

        With `overwrite`, a contiguous complex `b` is used as the work array and returned.
        """
        if overwrite and isinstance(b, np.ndarray) and b.dtype == np.complex128 and b.flags.c_contiguous and b.flags.writeable:
            x = b
        else:
            x = np.array(b, dtype=complex, order="C")
        if self._library.plu_solve_fmk(self._handle, x.ctypes.data, x.ctypes.data, x.shape[2], int(trans == "T")):
            raise MemoryError("native solve: out of memory")
        return x

    def solve_rows(self, b, trans="N"):
        """The same for one right-hand side of shape (size, F), the layout the kernel works in.

        `b` must be a contiguous complex array; it is overwritten. Returns a new array.
        """
        x = np.empty((self.size, self.F), complex)
        self._library.plu_solve(self._handle, b.ctypes.data, x.ctypes.data, int(trans == "T"))
        return x

    def solve_scattered(self, sides, trans="N"):
        """
        Like `_CondensedLU.solve_scattered`. The kernel works on arrays with one row per unknown,
        so nothing is transposed on the way in or out, and one work array serves every
        right-hand side (the kernel skips its zero rows).
        """
        work = self._work
        if work is None:
            work = self._work = np.empty((self.size, self.F), complex)
        solutions = []
        for parts in sides:
            work.fill(0.0)
            for rows, values in parts:
                work[rows] += values
            solutions.append(self.solve_rows(work, trans))
        return solutions


class _SingleLU:
    """
    A `_PatternLU` of a single matrix (the carrier system) with the `solve` of a SuperLU
    factorization: right-hand side and solution have shape (size,).
    """

    def __init__(self, factors):
        self.factors = factors

    def solve(self, b, trans="N"):
        return self.factors.solve_rows(np.array(b, dtype=complex).reshape(-1, 1), trans)[:, 0]


def forward(plan, carrier_entries, signal_entries, noise_entries):
    """Carrier (n,), signal (m, F) and noise (F,) outputs plus the adjoint helpers `Yc` and `MU`.

    Per frequency this takes one factorization and three solves: S^T conj(Y) = conj(w), then
    S X = rhs and S MU = (N + N^H) Y / 2. Simple unknowns are eliminated exactly beforehand and the
    rest goes through SuperLU as one block-diagonal system for all frequencies (`_CondensedLU`),
    or everything goes through the native kernel (`_PatternLU`). Everything between the solves is
    evaluated for all frequencies at once.

    The helpers have the layout of `signal`, one row per unknown and one column per frequency:
    `Yc` is conj(Y) with Y = S^-H w, which is what the transposed solve returns and what the
    reverse pass multiplies with, and `MU` is the second solution. All right-hand sides are zero
    outside a few rows and are handed over that way (`solve_scattered`), so that each solver
    builds them in the layout it works in.

    The last return value (carrier factorization, placing pre-values, signal factorization) is
    what `backward` would otherwise compute a second time.
    """
    carrier_lu, carrier_rhs = plan.carrier_matrix(carrier_entries)
    carrier = carrier_lu.solve(carrier_rhs)
    # The matrix values are built in the array the factorization works in. The factorization
    # overwrites them, so the right-hand side column is taken out first.
    spare = plan.spare
    V, pre = plan.signal_values(signal_entries, carrier, spare=spare)
    w = plan.detector_weights(carrier)
    N = plan.noise_values(noise_entries)
    source = V[~plan.in_matrix]
    factors = plan.signal_factors(V, overwrite=True, restore=lambda: plan.signal_values(signal_entries, carrier, spare=spare)[0])
    del V
    # S^-H w = conj(S^-T conj(w)); the weights are zero outside the detector ports.
    read = np.nonzero(w)[0]
    (Yc,) = factors.solve_scattered([[(read, np.conj(w[read])[:, None])]], trans="T")
    # N y and N^H y have few entries. They are the second right-hand side (N + N^H) y / 2, and
    # the noise power y^H N y is summed over the noise entries instead of over all unknowns.
    from_rows = Yc[plan.noise_rows]  # conj(y) at the row of every noise entry, shape (entries, F)
    products = np.conj(Yc[plan.noise_cols])  # y at its column
    np.multiply(N, products, out=products)  # the terms of N y
    power = np.einsum("ef,ef->f", from_rows, products).real
    half = []
    order, starts, slots = plan.noise_by_row
    if order.size:
        half.append((slots, 0.5 * _group_sum(products, order, starts)))
    order, starts, slots = plan.noise_by_col
    if order.size:
        np.conjugate(N, out=N)  # N is not used again
        np.conjugate(from_rows, out=from_rows)  # now y at the row, the einsum above is done
        N *= from_rows  # the terms of N^H y
        half.append((slots, 0.5 * _group_sum(N, order, starts)))
    signal, MU = factors.solve_scattered([[(plan.rhs_rows, source)], half])
    noise = np.sqrt(2 * power * NOISE_SCALE)
    return carrier, signal, noise, Yc, MU, (carrier_lu, pre, factors)


class _ReverseLayout:
    """
    The signal matrix entries the reverse pass visits, and what it needs from each of them.

    Only entries written by component functions or scaled by the carrier pass a cotangent on;
    the constant entries of the pattern are never visited. Of the written entries the caller reads
    the ones in `wanted` (the entries with a design variable behind them, see `_EntryBlock`).
    They form the first rows of the block of cotangents, in the order given. An entry that is
    scaled by the carrier also contributes to the cotangent of the carrier, through the sum over
    the frequencies of its cotangent times its value before the scaling. For an entry that nobody
    reads this sum is all that is needed. If its value is the same at every frequency (`varying`
    is False for it, or it is a constant of the pattern), only the sum of its cotangent is formed
    (`sum_*`), otherwise it gets a row at the end of the block.
    """

    def __init__(self, plan, wanted, varying):
        position, live = plan.update_positions, plan.update_live
        entries = position.size
        if wanted is None:
            wanted = np.arange(entries)
        if varying is None:
            varying = np.ones(entries, bool)
        self.count = count = wanted.size
        block_row = np.full(entries, -1)
        block_row[wanted] = np.arange(count)
        self.dead = np.nonzero(~live[wanted])[0]  # overwritten by a later write: no cotangent
        writer = np.full(plan.size, -1)  # the entry that sets each pattern position
        writer[position[live]] = np.nonzero(live)[0]
        scaled = np.nonzero(plan.placing_live)[0]  # placing entries: pattern value = -value * carrier
        source = writer[plan.placing_positions[scaled]]
        written = source >= 0
        in_block = written & (block_row[source] >= 0)
        extra = written & ~in_block & varying[source]
        summed = ~in_block & ~extra

        positions = np.concatenate([position[wanted], plan.placing_positions[scaled[extra]]])
        self.rows, cols = plan.rows[positions], plan.cols[positions]
        self.right_hand_side = np.nonzero(cols == plan.m)[0]
        self.cols = np.minimum(cols, plan.m - 1)
        self.conjugated = plan.conjugated[positions]
        self.placed = np.concatenate([block_row[source[in_block]], count + np.arange(int(extra.sum()))])
        self.placed_entries = np.concatenate([scaled[in_block], scaled[extra]])
        self.placed_carrier = plan.signal_carrier_indices[self.placed_entries]

        positions = plan.placing_positions[scaled[summed]]
        self.sum_rows, cols = plan.rows[positions], plan.cols[positions]
        self.sum_right_hand_side = cols == plan.m
        self.sum_cols = np.minimum(cols, plan.m - 1)
        self.sum_conjugated = plan.conjugated[positions]
        self.sum_entries = scaled[summed]
        self.sum_carrier = plan.signal_carrier_indices[self.sum_entries]


def _signal_adjoint(plan, carrier, signal, Yc, MU, signal_bar, g, V, factors):
    """
    The cotangent of the signal matrices as a sum of products: entry (r, c) of S_f gets
    sum_k left_k[r, f] * right_k[c, f], with arrays of shape (m, F). Also returns lam = S^-T
    signal_bar as a function of the rows wanted, shape (rows, F).

    The complete expression is -lam[r] signal[c] - 2 g conj(Y[r]) MU[c]. A sensitivity (noise
    over signal at the same detector) puts a cotangent on the signal that is a multiple alpha_f
    of conj(w) at every frequency, because signal and noise are read through the same weights w.
    Then lam_f = alpha_f conj(Y_f) without another solve, and the two products share their left
    factor: conj(Y[r]) * -(alpha signal[c] + 2 g MU[c]). `Yc` is conj(Y) and, like `MU`, already
    has the shape (m, F) (see `forward`).

    Any other cotangent takes one more solve with the factorization of the forward pass
    (`factors`), or with a new one if that is gone.
    """
    right = MU * (-2.0 * g)
    weights = np.conj(plan.detector_weights(carrier))
    read = np.nonzero(weights)[0]
    support = np.nonzero(np.any(signal_bar != 0, axis=1))[0]
    if read.size and np.isin(support, read).all():
        cotangent, weights = signal_bar[read], weights[read]
        alpha = (np.conj(weights) @ cotangent) / (np.conj(weights) @ weights).real
        rest = np.abs(cotangent - weights[:, None] * alpha).max(axis=0)
        if (rest <= PARALLEL_TOLERANCE * np.abs(cotangent).max(axis=0)).all():
            for start in range(0, right.shape[0], _ROW_BLOCK):
                right[start : start + _ROW_BLOCK] -= signal[start : start + _ROW_BLOCK] * alpha
            return [(Yc, right)], lambda rows: Yc[rows] * alpha
    if factors is None:
        factors = plan.signal_factors(V)
    lam = factors.solve_rows(np.array(signal_bar, dtype=complex, order="C"), trans="T")
    return [(lam, -signal), (Yc, right)], lambda rows: lam[rows]


def backward(
    plan,
    carrier_entries,
    signal_entries,
    carrier,
    signal,
    noise,
    Yc,
    MU,
    carrier_bar,
    signal_bar,
    noise_bar,
    kept=None,
    wanted=None,
    varying=None,
):
    """Cotangents of the carrier, signal and noise entries (JAX's bilinear convention).

    `signal`, `signal_bar` and the adjoint helpers `Yc` and `MU` of `forward` have shape (m, F).
    `kept` is the last return value of `forward` for the same inputs, when it is still available.

    The cotangent of the signal entries has one row per entry in `wanted` (default: every entry,
    in order). `varying` marks the signal entries whose value differs between the frequencies
    (default: all of them). See `_ReverseLayout` for what this saves. The work on the signal
    matrix entries runs entry by entry, on arrays of shape (entries, F).
    """
    n, m = plan.n, plan.m
    if kept is None:
        carrier_lu, _ = plan.carrier_matrix(carrier_entries)
        V, pre = plan.signal_values(signal_entries, carrier)
        factors = None
    else:
        (carrier_lu, pre, factors), V = kept, None
    F = signal.shape[1]
    layout = plan.reverse_layout(wanted, varying)
    carrier_bar = carrier_bar.astype(complex).copy()
    g = noise_bar * NOISE_SCALE / noise  # d noise / d power
    terms, lam_rows = _signal_adjoint(plan, carrier, signal, Yc, MU, signal_bar, g, V, factors)

    # Cotangents of the visited matrix entries, built in place: sum_k left_k[rows] * right_k[cols].
    rows, cols = layout.rows, layout.cols
    block = np.empty((rows.size, F), complex)
    for start in range(0, rows.size, _ROW_BLOCK):  # a block of rows at a time, see `_ROW_BLOCK`
        part, r, c = block[start : start + _ROW_BLOCK], rows[start : start + _ROW_BLOCK], cols[start : start + _ROW_BLOCK]
        for k, (left, right) in enumerate(terms):
            if k == 0:
                np.take(left, r, axis=0, out=part, mode="clip")  # "clip" writes straight into `out`
                part *= right[c]
            else:
                more = left[r]
                more *= right[c]
                part += more
    if layout.right_hand_side.size:  # entries of the right-hand side column
        block[layout.right_hand_side] = lam_rows(rows[layout.right_hand_side])
    if layout.conjugated.any():
        np.negative(block.imag, out=block.imag, where=layout.conjugated[:, None])  # conjugate those rows in place
    # Entries scaled by the carrier: value = -pre * carrier.
    placed, sources = layout.placed, layout.placed_carrier
    placed_bar = block[placed]
    carrier_parts = [np.einsum("ef,ef->e", placed_bar, pre[layout.placed_entries])]
    placed_bar *= -carrier[sources, None]
    block[placed] = placed_bar
    if layout.sum_rows.size:  # the same for entries that are equal at every frequency and that nobody reads
        sums = sum(np.einsum("ef,ef->e", left[layout.sum_rows], right[layout.sum_cols]) for left, right in terms)
        if layout.sum_right_hand_side.any():
            sums[layout.sum_right_hand_side] = lam_rows(layout.sum_rows[layout.sum_right_hand_side]).sum(axis=1)
        sums = np.where(layout.sum_conjugated, np.conj(sums), sums)
        carrier_parts.append(sums * pre[layout.sum_entries, 0])
        sources = np.concatenate([sources, layout.sum_carrier])
    np.subtract.at(carrier_bar, sources, np.concatenate(carrier_parts))
    block[layout.dead] = 0.0
    signal_entries_bar = block[: layout.count]

    written = plan.noise_base.size  # the entries after the constant ones come from noise functions
    noise_entries_bar = np.zeros((plan.noise_live.size, F), complex)
    noise_entries_bar[plan.noise_live] = g * Yc[plan.noise_rows[written:]] * np.conj(Yc[plan.noise_cols[written:]])

    w_bar = 2 * np.conj(MU @ g)  # sum over the frequencies of 2 g conj(MU)
    s = plan.selection
    carrier_bar += w_bar[:n] * s[:n] + np.conj(w_bar[n : 2 * n]) * s[n : 2 * n]
    lam_c = carrier_lu.solve(carrier_bar, trans="T")
    r, c = plan.cmi
    carrier_entries_bar = np.where(c == n, lam_c[r], -lam_c[r] * carrier[np.minimum(c, n - 1)])
    carrier_entries_bar = np.where(plan.carrier_live, carrier_entries_bar, 0)
    return carrier_entries_bar, signal_entries_bar, noise_entries_bar


def selection_cotangent(carrier, noise, MU, noise_bar):
    """
    Cotangent of the noise detector's selection vector, shape (m,). The detector weights are
    w = selection * carrier on the upper sideband, conj(selection * carrier) on the lower one and
    the selection itself elsewhere (`SystemPlan.detector_weights`); their cotangent is the
    `w_bar` of `backward`. `MU` has shape (m, F).
    """
    n = carrier.shape[0]
    selection_bar = 2 * np.conj(MU @ (noise_bar * NOISE_SCALE / noise))
    selection_bar[:n] *= carrier
    selection_bar[n : 2 * n] = np.conj(selection_bar[n : 2 * n]) * carrier
    return selection_bar


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
    "qhd_parameter_indices",
    "qhd_placing_indices",
    "linked_indices",
    "linking_function_indices",
    "indices_to_link",
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


def _links_supported(arrays):
    """Parameter links are covered when their functions are known at trace time and in `LINK_RULES`."""
    if arrays["linked_indices"].shape[0] == 0:
        return True
    indices = arrays["linking_function_indices"]
    if not isinstance(indices, np.ndarray):  # a traced index array: the functions are not known here
        return False
    from differometor.build import LINKING_FUNCTIONS

    return all(0 <= i < len(LINKING_FUNCTIONS) and _link_rule(LINKING_FUNCTIONS[i]) is not None for i in indices.reshape(-1).tolist())


def supports(arrays):
    """
    Static check (shapes, dtypes and the linking functions) whether the fast path covers these
    `simulate` arguments: exactly one quantum noise detector (a `qnoised` or a balanced homodyne
    detector `qhd`), a single carrier operating point, real design variables and only parameter
    links whose functions are in `LINK_RULES`.
    """
    optimized = arrays["optimized_parameters"]
    return (
        _links_supported(arrays)
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


# Work of the forward pass that the backward pass can reuse: both plans, the matrix entries
# with their Jacobian tape, the placing pre-values, both factorizations, and the results
# with the adjoint helpers Y and MU. With the cache on, a forward pass that is followed by a
# pullback leaves all of this on the host and hands JAX only the outputs and a key; the pullback
# finds its forward pass through the key. Nothing depends on the entry still being there: when
# it is gone the backward callback runs the forward pass again from its own inputs.
#
# With `FORWARD_CACHE = False` (read when a function is traced) the callbacks keep no state:
# Y and MU flow through JAX as residuals and the backward callback rebuilds the rest.
FORWARD_CACHE = True
_FORWARD_CACHE_SIZE = 2
_forward_cache = {}


def _run_forward(optimized_parameters, host_arrays, taped):
    """The forward pass on the host: results (carrier, signal, noise, Y, MU) and the reusable work."""
    A = dict(zip(HOST_KEYS, host_arrays))
    entry_plan, plan = EntryPlan(A), SystemPlan(A)
    plan.selection = entry_plan.selection(optimized_parameters)  # with the phase of a homodyne detector
    # The signal entries are only needed inside the value store of the signal matrices. When every
    # one of them has a row of its own there, they are written straight into it.
    target = stored = None
    if plan.update_live.all():
        F, point = entry_plan.signal_changing_values.shape[1], np.array(optimized_parameters, float)

        def fill():
            store = plan.signal_store(F, plan.spare)
            entry_plan.signal_into(point, (store, plan.update_positions))
            return store

        target = (plan.signal_store(F, plan.spare), plan.update_positions)
        stored = _StoredEntries(target[0], fill)
    if taped:  # a pullback will follow: evaluate the entries with derivatives once, here
        entries, tape = entry_plan.entries(optimized_parameters, target)
    else:
        entries, tape = entry_plan.values(optimized_parameters, target), None
    if stored is not None:
        entries = (entries[0], stored, entries[2])
    *results, kept = forward(plan, *entries)
    return tuple(results), (entry_plan, plan, entries, tape, kept)


def _run_backward(optimized_parameters, results, work, carrier_bar, signal_bar, noise_bar):
    """The backward pass on the host, from the results and the work of the forward pass."""
    carrier, signal, noise, Yc, MU = results
    entry_plan, plan, entries, tape, kept = work
    F = signal.shape[1]
    noise_bar = np.asarray(noise_bar, float).reshape(F)
    entries_bar = backward(
        plan,
        entries[0],
        entries[1],
        carrier,
        signal,
        noise,
        Yc,
        MU,
        np.asarray(carrier_bar)[:, 0],
        np.asarray(signal_bar),
        noise_bar,
        kept=kept,
        wanted=entry_plan.signal.wanted,  # only the entries the pullback reads
        varying=entry_plan.signal.entry_varying,
    )
    selection_bar = None
    if entry_plan.phase_is_design_variable:
        selection_bar = selection_cotangent(carrier, np.asarray(noise).reshape(F), MU, noise_bar)
    gradient = entry_plan.pullback(
        tape, *entries_bar, value_count=optimized_parameters.shape[0], compact=True, optimized_parameters=optimized_parameters, selection_bar=selection_bar
    )
    return gradient.astype(optimized_parameters.dtype)


def _host_forward(optimized_parameters, *host_arrays):
    """Stateless forward callback: the outputs and the adjoint helpers Y and MU."""
    (carrier, signal, noise, Y, MU), _ = _run_forward(optimized_parameters, host_arrays, taped=False)
    return carrier[:, None], signal, noise.reshape(-1), Y, MU


def _host_forward_plain(optimized_parameters, *host_arrays):
    """Forward callback when no pullback follows: only the outputs."""
    return _host_forward(optimized_parameters, *host_arrays)[:3]


def _host_forward_cached(optimized_parameters, *host_arrays):
    """Forward callback in front of a pullback: the outputs and the key of the cache entry."""
    results, work = _run_forward(optimized_parameters, host_arrays, taped=True)
    token = int.from_bytes(os.urandom(6), "little") + 1  # exact in float64, never reused in practice
    while len(_forward_cache) >= _FORWARD_CACHE_SIZE:
        del _forward_cache[next(iter(_forward_cache))]
    _forward_cache[token] = (np.array(optimized_parameters, float), results, work)
    carrier, signal, noise = results[:3]
    return carrier[:, None], signal, noise.reshape(-1), np.float64(token)


def _host_backward(optimized_parameters, *rest):
    """Stateless backward callback: everything comes from the residuals."""
    k = len(HOST_KEYS)
    carrier, signal, noise, Y, MU, carrier_bar, signal_bar, noise_bar = rest[k:]
    A = dict(zip(HOST_KEYS, rest[:k]))
    entry_plan, plan = EntryPlan(A), SystemPlan(A)
    plan.selection = entry_plan.selection(optimized_parameters)
    entries, tape = entry_plan.entries(optimized_parameters)
    results = (np.asarray(carrier)[:, 0], np.asarray(signal), np.asarray(noise).reshape(-1), np.asarray(Y), np.asarray(MU))
    return _run_backward(optimized_parameters, results, (entry_plan, plan, entries, tape, None), carrier_bar, signal_bar, noise_bar)


def _host_backward_cached(optimized_parameters, *rest):
    """Backward callback for `_host_forward_cached`: reuses the forward pass, or repeats it if it is gone."""
    k = len(HOST_KEYS)
    token, carrier_bar, signal_bar, noise_bar = rest[k:]
    cached = _forward_cache.get(int(token))
    if cached is not None and np.array_equal(cached[0], optimized_parameters):
        _, results, work = cached
    else:
        results, work = _run_forward(optimized_parameters, rest[:k], taped=True)
    return _run_backward(optimized_parameters, results, work, carrier_bar, signal_bar, noise_bar)


def _result_shapes(host):
    """Shapes of the forward outputs and of the adjoint helpers, plus the shape of `noise`."""
    n, m, F, noise_shape = _output_shapes(host)
    outputs = (
        jax.ShapeDtypeStruct((n, 1), jnp.complex128),
        jax.ShapeDtypeStruct((m, F), jnp.complex128),
        jax.ShapeDtypeStruct((F,), jnp.float64),
    )
    helpers = (jax.ShapeDtypeStruct((m, F), jnp.complex128), jax.ShapeDtypeStruct((m, F), jnp.complex128))
    return outputs, helpers, noise_shape


def _call_host(function, shapes, optimized_parameters, host, *more):
    return jax.pure_callback(
        function, shapes, optimized_parameters, *(host[k] for k in HOST_KEYS), *more, vmap_method="sequential"
    )


def _is_primal(x):
    return isinstance(x, jax.custom_derivatives.CustomVJPPrimal)


def _zeros_if_symbolic(value):
    if isinstance(value, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(value.shape, value.dtype)
    return value


@jax.custom_vjp
def _fast(optimized_parameters, host, others):
    outputs, _, noise_shape = _result_shapes(host)
    carrier, signal, noise = _call_host(_host_forward_plain, outputs, optimized_parameters, host)
    return carrier, signal, noise.reshape(noise_shape)


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
    outputs, helpers, noise_shape = _result_shapes(host)
    # Under `vmap` all forward passes of a batch run before the first pullback, so the small
    # cache would only serve the last ones. Batched calls therefore take the stateless route.
    # (The tracer class has no public name any more; if this check stops matching, results stay
    # correct and a batch only gets slower.)
    batched = any(type(leaf).__name__ == "BatchTracer" for leaf in jax.tree_util.tree_leaves((p, host)))
    if FORWARD_CACHE and not batched:
        key = jax.ShapeDtypeStruct((), jnp.float64)
        carrier, signal, noise, token = _call_host(_host_forward_cached, outputs + (key,), p, host)
        return (carrier, signal, noise.reshape(noise_shape)), (p, host, token)
    carrier, signal, noise, Y, MU = _call_host(_host_forward, outputs + helpers, p, host)
    return (carrier, signal, noise.reshape(noise_shape)), (p, host, carrier, signal, noise, Y, MU)


def _fast_bwd(residuals, cotangents):
    cotangents = tuple(_zeros_if_symbolic(c) for c in cotangents)
    if len(residuals) == 1:
        return residuals[0](cotangents)
    p, host, *rest = residuals
    function = _host_backward_cached if len(rest) == 1 else _host_backward
    gradient = _call_host(function, jax.ShapeDtypeStruct(p.shape, p.dtype), p, host, *rest, *cotangents)
    return gradient, None, None


_fast.defvjp(_fast_fwd, _fast_bwd, symbolic_zeros=True)


def simulate_fast(arrays):
    """Fast `simulate` for argument dicts accepted by `supports`."""
    host = {key: arrays[key] for key in HOST_KEYS}
    others = {key: value for key, value in arrays.items() if key not in HOST_KEYS and key != "optimized_parameters"}
    return _fast(arrays["optimized_parameters"], host, others)
