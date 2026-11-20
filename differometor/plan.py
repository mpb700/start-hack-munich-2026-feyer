"""
Static build plan for `simulate`.

`benchmark.py` passes every build array into `jax.jit` as a traced argument, so
`simulate` cannot see at trace time which component function belongs to which
matrix entry. It therefore evaluates all component functions through a batched
`lax.switch`, which XLA turns into "evaluate every branch, then select".

A `BuildPlan` carries that structure as static pytree aux data instead (it has
no array leaves). With it, `simulate` can:
- evaluate each component function once on exactly the entries that use it,
- evaluate components whose inputs are not swept only once instead of per row,
- conjugate lower-sideband entries on the value list instead of on dense blocks,
- write every matrix entry exactly once (duplicate writes are resolved to the
  last writer at build time, matching the sequential `.at[].set` semantics).

All values are still computed from the traced parameters on every call; only the
index structure is fixed. A plan belongs to the build it was made from.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np


def _last_writer(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Positions (in original order) of the last write to each distinct (row, col)."""
    keys = rows.astype(np.int64) * (int(cols.max(initial=0)) + 1) + cols.astype(np.int64)
    _, first_in_reversed = np.unique(keys[::-1], return_index=True)
    return np.sort(len(keys) - 1 - first_in_reversed)


@dataclass(frozen=True, eq=False)
class EntryUpdate:
    """Static description of how component functions fill one matrix."""

    inputs: np.ndarray  # (K, W) parameter indices per component
    invariant_groups: tuple  # ((function_index, component_positions), ...) evaluated on row 0
    varying_groups: tuple  # same, evaluated on every row
    invariant_entries: tuple  # (slot, output_column, row, column) arrays, one write each
    varying_entries: tuple


def _entry_update(function_input_indices, function_indices, output_indices, matrix_indices, shape, changing):
    inputs = np.asarray(function_input_indices, dtype=np.int64)
    functions = np.asarray(function_indices, dtype=np.int64)
    output_rows, output_cols = (np.asarray(a, dtype=np.int64) for a in output_indices)
    rows = np.asarray(matrix_indices[0], dtype=np.int64) % shape[0]
    cols = np.asarray(matrix_indices[1], dtype=np.int64) % shape[1]

    keep = _last_writer(rows, cols)
    output_rows, output_cols, rows, cols = output_rows[keep], output_cols[keep], rows[keep], cols[keep]

    # A component is row-invariant if none of its inputs is a swept parameter.
    invariant = ~np.isin(inputs, np.asarray(changing, dtype=np.int64)).any(axis=1)

    slot = np.empty(len(functions), dtype=np.int64)
    groups = {True: [], False: []}
    counts = {True: 0, False: 0}
    for flag in (True, False):
        for function in np.unique(functions[invariant == flag]):
            positions = np.flatnonzero((functions == function) & (invariant == flag))
            groups[flag].append((int(function), positions))
            slot[positions] = counts[flag] + np.arange(len(positions))
            counts[flag] += len(positions)

    entries = {}
    for flag in (True, False):
        mask = invariant[output_rows] == flag
        entries[flag] = (slot[output_rows[mask]], output_cols[mask], rows[mask], cols[mask])

    return EntryUpdate(
        inputs=inputs,
        invariant_groups=tuple(groups[True]),
        varying_groups=tuple(groups[False]),
        invariant_entries=entries[True],
        varying_entries=entries[False],
    )


def _sideband_conjugation_mask(rows, cols, carrier_size):
    """True where `simulate` conjugates the lower-sideband blocks of the signal matrix."""
    cs = carrier_size
    lower_rows = (rows >= cs) & (rows < 2 * cs)
    return (lower_rows & (cols >= cs) & (cols < 2 * cs)) | ((rows >= 2 * cs) & (cols < cs)) | (lower_rows & (cols >= 2 * cs))


@dataclass(frozen=True, eq=False)
class BuildPlan:
    carrier: EntryUpdate
    signal: EntryUpdate
    noise: EntryUpdate
    signal_invariant_conjugate: np.ndarray  # bool mask per invariant signal entry
    signal_varying_conjugate: np.ndarray
    placing: tuple  # (rows, cols, carrier_indices, conjugate_mask) for carrier-scaled signal entries
    noise_base: tuple  # (rows, cols) of constant noise-matrix entries not overwritten by functions


jax.tree_util.register_pytree_node(BuildPlan, lambda plan: ((), plan), lambda plan, _: plan)


def make_plan(prepared_arrays, matrices, carrier_arrays, signal_arrays, noise_arrays) -> BuildPlan:
    (_, _, _, carrier_changing, carrier_values, signal_changing, signal_values) = prepared_arrays
    _, carrier_matrix, signal_matrix, noise_matrix, *_ = matrices

    # A parameter only differs between rows if its sweep has more than one value.
    # (Without a sweep, `prepare_arrays` puts a placeholder index with a single
    # value here; index 0 is also the padding of unused function inputs.)
    def swept(indices, values):
        return np.asarray(indices, dtype=np.int64) if np.shape(values)[-1] > 1 else np.zeros(0, dtype=np.int64)

    carrier_changing = swept(carrier_changing, carrier_values)
    swept = np.union1d(carrier_changing, swept(signal_changing, signal_values))
    carrier_size = carrier_matrix.shape[0]

    carrier = _entry_update(*carrier_arrays, carrier_matrix.shape, carrier_changing)
    signal = _entry_update(*signal_arrays[:4], signal_matrix.shape, swept)
    noise = _entry_update(*noise_arrays, noise_matrix.shape, swept)

    placing_rows = np.asarray(signal_arrays[5][0], dtype=np.int64) % signal_matrix.shape[0]
    placing_cols = np.asarray(signal_arrays[5][1], dtype=np.int64) % signal_matrix.shape[1]
    placing_carrier = np.asarray(signal_arrays[4], dtype=np.int64).reshape(-1)
    keep = _last_writer(placing_rows, placing_cols)
    placing_rows, placing_cols, placing_carrier = placing_rows[keep], placing_cols[keep], placing_carrier[keep]

    written = set(zip(*noise.invariant_entries[2:], strict=True)) | set(zip(*noise.varying_entries[2:], strict=True))
    base_rows, base_cols = np.nonzero(np.asarray(noise_matrix))
    unwritten = np.array([(r, c) not in written for r, c in zip(base_rows, base_cols)], dtype=bool)

    return BuildPlan(
        carrier=carrier,
        signal=signal,
        noise=noise,
        signal_invariant_conjugate=_sideband_conjugation_mask(*signal.invariant_entries[2:], carrier_size),
        signal_varying_conjugate=_sideband_conjugation_mask(*signal.varying_entries[2:], carrier_size),
        placing=(placing_rows, placing_cols, placing_carrier, _sideband_conjugation_mask(placing_rows, placing_cols, carrier_size)),
        noise_base=(base_rows[unwritten], base_cols[unwritten]),
    )
