"""Evaluation metrics for Page-EntroKV.

The centrepiece is the *Unique Occupancy Ratio* (UOR).  For a multi-layer page
table in which every selected physical block :math:`P_b` is assigned to at least
one KV-head slot :math:`(l, g)`, define its occupancy

.. math::
    \\text{occ}(P_b) = \\#\\{(l, g) : P_b \\in \\mathcal{S}_{l,g}\\}.

The UOR is the fraction of occupied blocks that are occupied by *exactly one*
slot:

.. math::
    \\text{UOR} = \\frac{|\\{P_b : \\text{occ}(P_b) = 1\\}|}
                     {|\\{P_b : \\text{occ}(P_b) \\ge 1\\}|}.

Page-EntroKV's selection (``paged_mapping.select_top_k_per_group``) enforces
:math:`\\text{UOR} \\equiv 1.00`; this module provides the metric plus related
diagnostics (coverage, compaction, eviction recall) used by the paper and the
integration benchmarks.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt


def occupancy_counts(
    selected: Union[
        npt.ArrayLike,
        Sequence[Sequence[int]],
        Sequence[npt.NDArray[np.int64]],
    ],
    num_blocks: int,
) -> npt.NDArray[np.int64]:
    """Count how many slots selected each physical block.

    Parameters
    ----------
    selected:
        Either a 1-D flat array (a single slot) or a sequence of per-slot block
        index arrays (one entry per KV group / layer).
    num_blocks:
        Total number of physical blocks :math:`\\mathcal{B}_{\\text{total}}`.

    Returns
    -------
    ndarray
        ``counts[b]`` is the number of slots that selected block ``b``.
    """
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
    counts = np.zeros(num_blocks, dtype=np.int64)

    def _add(indices: npt.NDArray[np.int64]) -> None:
        if indices.size == 0:
            return
        if np.any(indices < 0) or np.any(indices >= num_blocks):
            raise IndexError(
                f"block index out of range [0, {num_blocks}); got {indices}"
            )
        np.add.at(counts, indices, 1)

    if _is_nested_selection(selected):
        for slot in selected:
            _add(np.asarray(slot, dtype=np.int64).ravel())
    else:
        _add(np.asarray(selected, dtype=np.int64).ravel())
    return counts


def _is_nested_selection(selected) -> bool:
    """Return ``True`` iff ``selected`` is a sequence of per-slot sequences."""
    if isinstance(selected, np.ndarray):
        return bool(selected.dtype == object or selected.ndim >= 2)
    if not isinstance(selected, (list, tuple)):
        return False
    if len(selected) == 0:
        return False
    first = selected[0]
    return not isinstance(first, (int, np.integer))


def unique_occupancy_ratio(
    selected: Union[
        npt.ArrayLike,
        Sequence[Sequence[int]],
        Sequence[npt.NDArray[np.int64]],
    ],
    num_blocks: int,
) -> float:
    """Compute the Unique Occupancy Ratio (UOR).

    The UOR is the fraction of occupied blocks that are occupied by exactly one
    slot (see :func:`occupancy_counts`).  An empty selection vacuously satisfies
    "every occupied block is uniquely occupied", so it returns ``1.0``.
    Page-EntroKV's selection guarantees ``UOR == 1.0`` exactly.
    """
    counts = occupancy_counts(selected, num_blocks)
    occupied = int((counts >= 1).sum())
    if occupied == 0:
        return 1.0
    singletons = int((counts == 1).sum())
    return singletons / occupied


def coverage(
    selected: Union[npt.ArrayLike, Sequence[Sequence[int]]],
    num_blocks: int,
) -> float:
    """Fraction of physical blocks occupied by at least one slot."""
    counts = occupancy_counts(selected, num_blocks)
    if num_blocks <= 0:
        return 0.0
    return float((counts >= 1).sum()) / num_blocks


def compaction(
    selected: Union[npt.ArrayLike, Sequence[Sequence[int]]],
    num_blocks: int,
) -> float:
    """Selected blocks per occupied block (1.0 means zero duplication)."""
    counts = occupancy_counts(selected, num_blocks)
    occupied = int((counts >= 1).sum())
    if occupied == 0:
        return 0.0
    return float(int(counts.sum())) / occupied


def eviction_recall(
    selected_blocks: Sequence[int], oracle_blocks: Sequence[int]
) -> float:
    """Fraction of oracle (ground-truth) blocks that were selected."""
    oracle = set(oracle_blocks)
    if not oracle:
        return 1.0
    return len(set(selected_blocks) & oracle) / len(oracle)


def page_hit_rate(
    selected_blocks: Sequence[int], required_blocks: Sequence[int]
) -> float:
    """Fraction of required blocks present in the selected set."""
    return eviction_recall(selected_blocks, required_blocks)


def retention_ratio(
    pinned_blocks: Sequence[int], selected_blocks: Sequence[int]
) -> float:
    """Fraction of pinned anchor pages that were actually selected/retained."""
    pinned = set(pinned_blocks)
    if not pinned:
        return 1.0
    return len(pinned & set(selected_blocks)) / len(pinned)


def resident_token_mask(
    selected_blocks: Sequence[int], block_size: int, num_tokens: int
) -> npt.NDArray[np.bool_]:
    """Token-level keep mask implied by a set of resident blocks.

    Token ``t`` is kept iff its page ``t // block_size`` is in
    ``selected_blocks``.  This is the inverse of the block reduction: it maps a
    page selection back onto the token axis so retention can be measured at
    token granularity.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    mask = np.zeros(num_tokens, dtype=bool)
    for block in selected_blocks:
        block = int(block)
        if block < 0:
            raise ValueError(f"block index must be non-negative, got {block}")
        start = block * block_size
        end = min(start + block_size, num_tokens)
        if start < num_tokens:
            mask[start:end] = True
    return mask


def token_retention(
    token_mask: npt.ArrayLike, block_size: int
) -> float:
    """Fraction of invariant tokens that fall inside a pinned page.

    Given the token-level invariant mask :math:`\\mathcal{T}_{\\text{AST}}` and
    the page size :math:`B`, returns the fraction of invariant tokens whose page
    also contains at least one invariant token (i.e. tokens that survive pinning).
    """
    mask = np.asarray(token_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"token_mask must be 1-D, got shape {mask.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    num_invariant = int(mask.sum())
    if num_invariant == 0:
        return 1.0
    num_tokens = mask.size
    num_blocks = int(math.ceil(num_tokens / block_size))
    block_density = np.zeros(num_blocks, dtype=np.int64)
    for block_index in range(num_blocks):
        start = block_index * block_size
        end = min(start + block_size, num_tokens)
        block_density[block_index] = int(mask[start:end].sum())
    pinned_pages = block_density > 0
    retained = 0
    for block_index in range(num_blocks):
        if pinned_pages[block_index]:
            start = block_index * block_size
            end = min(start + block_size, num_tokens)
            retained += int(mask[start:end].sum())
    return retained / num_invariant


__all__ = [
    "compaction",
    "coverage",
    "eviction_recall",
    "occupancy_counts",
    "page_hit_rate",
    "resident_token_mask",
    "retention_ratio",
    "token_retention",
    "unique_occupancy_ratio",
]
