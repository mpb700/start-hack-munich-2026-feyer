"""Direct SuperLU access without importing ``scipy``.

The fast host path only needs SuperLU's ``gstrf``/``solve``. Importing
``scipy.sparse.linalg`` pulls in ~150 modules (~0.14 s warm, ~0.76 s cold) and
is the largest avoidable cost in the measured first run. This module locates
scipy's compiled ``_superlu*.so`` through ``importlib.util.find_spec("scipy")``
(which does not execute the scipy package), loads it with an
``ExtensionFileLoader`` (~1-2 ms) and calls ``gstrf`` with exactly the same
arguments and options as :func:`scipy.sparse.linalg.splu` in scipy 1.18.1, so
factorisations and solutions are bit-identical.

If anything fails (scipy missing, extension not found, ABI mismatch, ``gstrf``
raises), :func:`splu` transparently falls back to
``scipy.sparse.linalg.splu``; that fallback is the only place that imports
scipy.
"""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os

import numpy as np

__all__ = ["CSC", "splu", "direct_available"]


class CSC:
    """Canonical CSC matrix given by explicit arrays (no ``scipy.sparse`` needed)."""

    __slots__ = ("data", "indices", "indptr", "shape")

    def __init__(self, data, indices, indptr, shape):
        self.data = np.ascontiguousarray(data)
        self.indices = np.ascontiguousarray(indices)
        self.indptr = np.ascontiguousarray(indptr)
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def nnz(self):
        return int(self.data.size)


class _CSCShim:
    """Stand-in for the CSC class that ``gstrf`` uses to materialise ``L``/``U`` lazily.

    The fast path only calls ``solve``; ``L``/``U`` are never touched.
    """

    __slots__ = ("_args", "_kwargs")

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs


class _LU:
    """Smallest wrapper around a SuperLU object that exposes the API the fast path uses."""

    __slots__ = ("_lu", "is_fallback")

    def __init__(self, lu, is_fallback):
        self._lu = lu
        self.is_fallback = is_fallback

    @property
    def perm_c(self):
        return self._lu.perm_c

    @property
    def perm_r(self):
        return self._lu.perm_r

    def solve(self, b, trans="N"):
        return self._lu.solve(b, trans=trans)


def _load_gstrf():
    """Load scipy's ``_superlu`` extension module without importing scipy."""
    try:
        spec = importlib.util.find_spec("scipy")
        if spec is None or not spec.submodule_search_locations:
            return None
        root = list(spec.submodule_search_locations)[0]
        pattern = os.path.join(root, "sparse", "linalg", "_dsolve", "_superlu*.so")
        paths = sorted(glob.glob(pattern))
        if not paths:
            return None
        loader = importlib.machinery.ExtensionFileLoader("_superlu", paths[0])
        module_spec = importlib.util.spec_from_loader("_superlu", loader)
        module = importlib.util.module_from_spec(module_spec)
        loader.exec_module(module)
        return getattr(module, "gstrf", None)
    except Exception:  # noqa: BLE001 - fallback to scipy is intentional
        return None


_gstrf = _load_gstrf()


def direct_available():
    """True when the direct SuperLU extension was loaded successfully."""
    return _gstrf is not None


def _fallback_splu(matrix, permc_spec):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu as scipy_splu

    data = np.ascontiguousarray(matrix.data)
    indices = np.ascontiguousarray(matrix.indices, dtype=np.intc)
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.intc)
    values = csc_matrix((data, indices, indptr), shape=matrix.shape)
    return _LU(scipy_splu(values, permc_spec=permc_spec), True)


def splu(matrix, permc_spec=None):
    """Factorise a canonical CSC matrix; options identical to scipy's ``splu``."""
    gstrf = _gstrf
    if gstrf is not None:
        # Exactly what scipy.sparse.linalg.splu (1.18.1) passes to gstrf.
        options = {
            "DiagPivotThresh": None,
            "ColPerm": permc_spec,
            "PanelSize": None,
            "Relax": None,
        }
        if permc_spec == "NATURAL":
            options["SymmetricMode"] = True
        indices = np.ascontiguousarray(matrix.indices, dtype=np.intc)
        indptr = np.ascontiguousarray(matrix.indptr, dtype=np.intc)
        try:
            direct = gstrf(
                int(matrix.shape[0]),
                int(matrix.data.size),
                matrix.data,
                indices,
                indptr,
                csc_construct_func=_CSCShim,
                ilu=False,
                options=options,
            )
            return _LU(direct, False)
        except Exception:  # noqa: BLE001 - fall back to scipy on any SuperLU problem
            pass
    return _fallback_splu(matrix, permc_spec)
