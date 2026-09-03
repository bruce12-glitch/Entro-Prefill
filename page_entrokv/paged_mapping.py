"""Paged-frame mapping: block-max reduction and unique-occupancy selection.

Theory
------
The pooled attention distribution :math:`A_{l,g}^{\\text{pooled}}(t)` of physical
KV head :math:`g` is projected onto discrete physical memory pages
:math:`P_b` of size :math:`B` via the Block-Max reduction

.. math::
    I_{l,g}(P_b) = \\max_{t \\in \\text{tokens}(P_b)}
        A_{l,g}^{\\text{pooled}}(t).

Anchor pages :math:`\\mathcal{P}_{\\text{pinned}}` are pinned by forcing

.. math::
    I_{l,g}(P_b) = +\\infty \\quad \\forall\\, P_b \\in \\mathcal{P}_{\\text{pinned}},

so they are always selected.  Each group then keeps the top

.. math::
    K_l = \\left\\lfloor \\mathcal{B}_l / (H_{\\text{KV}} \\cdot B) \\right\\rfloor

blocks by importance.  Selection enforces a *unique occupancy ratio* of exactly
:math:`\\text{UOR} \\equiv 1.00`: every selected physical page is occupied by at
most one selected KV-head slot, so the multi-layer page table contains no
duplicate blocks (see :func:`select_top_k_per_group` for the exact guarantee and
:mod:`page_entrokv.metrics` for the :math:`\\text{UOR}` definition).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_INF = float("inf")


def block_max_reduce(
    pooled_attention: npt.ArrayLike, block_size: int
) -> npt.NDArray[np.float64]:
    """Block-Max reduction of a distribution onto physical pages.

    Parameters
    ----------
    pooled_attention:
        Distribution of shape ``(T,)`` or ``(G, T)``.
    block_size:
        Tokens per page :math:`B`.

    Returns
    -------
    ndarray
        Page importances of shape ``(N,)`` or ``(G, N)`` where
        :math:`N = \\lceil T / B \\rceil`.  The final (partial) page is reduced
        over the tokens it actually contains.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    a = np.asarray(pooled_attention, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise ValueError(
            f"pooled_attention must be 1-D (T,) or 2-D (G, T), got shape {a.shape}"
        )
    if a.shape[-1] == 0:
        raise ValueError("pooled_attention has zero sequence length")

    num_tokens = a.shape[-1]
    num_blocks = int(np.ceil(num_tokens / block_size))
    single = a.ndim == 1
    reduced = np.zeros(
        (num_blocks,) if single else (a.shape[0], num_blocks), dtype=np.float64
    )
    for block_index in range(num_blocks):
        start = block_index * block_size
        end = min(start + block_size, num_tokens)
        segment = a[..., start:end]
        if segment.shape[-1]:
            reduced[..., block_index] = segment.max(axis=-1)
    return reduced


def block_sum_reduce(
    pooled_attention: npt.ArrayLike, block_size: int
) -> npt.NDArray[np.float64]:
    """Block-Sum reduction of a distribution onto physical pages.

    Provided for the ablation studies; production selection uses
    :func:`block_max_reduce`.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    a = np.asarray(pooled_attention, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise ValueError(
            f"pooled_attention must be 1-D (T,) or 2-D (G, T), got shape {a.shape}"
        )
    if a.shape[-1] == 0:
        raise ValueError("pooled_attention has zero sequence length")

    num_tokens = a.shape[-1]
    num_blocks = int(np.ceil(num_tokens / block_size))
    single = a.ndim == 1
    reduced = np.zeros(
        (num_blocks,) if single else (a.shape[0], num_blocks), dtype=np.float64
    )
    for block_index in range(num_blocks):
        start = block_index * block_size
        end = min(start + block_size, num_tokens)
        segment = a[..., start:end]
        if segment.shape[-1]:
            reduced[..., block_index] = segment.sum(axis=-1)
    return reduced


def pin_blocks(
    importances: npt.NDArray[np.float64],
    pinned_block_indices: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Force pinned anchor pages to :math:`+\\infty` importance.

    Parameters
    ----------
    importances:
        Page importances of shape ``(N,)`` or ``(G, N)``.
    pinned_block_indices:
        Block indices :math:`\\mathcal{P}_{\\text{pinned}}`.

    Returns
    -------
    ndarray
        A copy of ``importances`` with pinned columns set to ``+inf``.
    """
    scores = np.array(importances, dtype=np.float64, copy=True)
    pinned = np.asarray(pinned_block_indices, dtype=np.int64)
    if pinned.ndim != 1:
        raise ValueError(
            f"pinned_block_indices must be 1-D, got shape {pinned.shape}"
        )
    if scores.ndim not in (1, 2):
        raise ValueError(
            f"importances must be 1-D (N,) or 2-D (G, N), got shape {scores.shape}"
        )
    num_blocks = scores.shape[-1]
    if np.any(pinned < 0) or np.any(pinned >= num_blocks):
        raise IndexError(
            f"pinned block index out of range [0, {num_blocks}); got {pinned}"
        )
    if pinned.size:
        scores[..., pinned] = _INF
    return scores


def select_blocks(
    importances: npt.ArrayLike, k: int
) -> npt.NDArray[np.int64]:
    """Select the indices of the top-``k`` blocks by importance (one row).

    Ties are broken towards the lowest block index.  ``+inf`` (pinned) entries
    always precede finite ones.  Returns a sorted index array.
    """
    scores = np.asarray(importances, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"importances must be 1-D, got shape {scores.shape}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k > scores.size:
        raise ValueError(
            f"k={k} exceeds the number of blocks {scores.size}"
        )
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k].astype(np.int64))


def select_top_k_per_group(
    importances: npt.ArrayLike,
    k_per_group: Union[int, Sequence[int]],
    pinned_block_indices: Sequence[int],
) -> npt.NDArray[np.int64]:
    """Select blocks per group while enforcing :math:`\\text{UOR} \\equiv 1.00`.

    Selection separates the *permanent* anchor pages from the *dynamic* budget:

    * Anchor pages :math:`\\mathcal{P}_{\\text{pinned}}` are pinned by forcing
      :math:`I_{l,g}(P_b) = +\\infty` and are **always** returned (permanent
      anchors never fall victim to the dynamic budget).
    * Each KV group additionally selects its top :math:`K_l` *dynamic* blocks
      (anchor pages excluded from the dynamic pool) with no block ever selected
      by two groups.

    Parameters
    ----------
    importances:
        Page importances of shape ``(G, N)``.
    k_per_group:
        Integer budget per group, or a length-``G`` sequence.
    pinned_block_indices:
        Anchor pages to pin with :math:`+\\infty` importance.

    Returns
    -------
    PagedSelection
        The per-layer selection: ``selected_blocks`` is the sorted union of the
        pinned anchor pages and the per-group dynamic selections, while
        ``per_group_selected`` exposes each group's dynamic blocks.

    Unique-Occupancy guarantee
    --------------------------
    Rows are processed from the most to the least focused group (smallest
    collision entropy first) so focused groups are never starved.  A group's
    already-selected dynamic blocks are masked before its own top-``K``
    selection, so **no dynamic block is selected by more than one group**.
    Letting :math:`\\mathcal{D}` be the set of dynamically selected blocks and
    :math:`\\text{occ}(P_b) = \\#\\{g : P_b \\in \\mathcal{D}_g\\}`,

    .. math::
        \\max_{P_b \\in \\mathcal{D}} \\text{occ}(P_b) \\le 1.

    The per-group selections are therefore pairwise disjoint, so the Unique
    Occupancy Ratio (see :func:`page_entrokv.metrics.unique_occupancy_ratio`,
    computed over ``per_group_selected``) equals exactly
    :math:`\\text{UOR} = 1.00`: every dynamically occupied page is occupied by
    exactly one group, none by two or more.  Anchor pages are shared by design
    and are excluded from the duplicate accounting.
    """
    scores = np.asarray(importances, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(
            f"importances must be 2-D (G, N), got shape {scores.shape}"
        )
    num_groups, num_blocks = scores.shape
    if num_blocks == 0:
        raise ValueError("importances has zero blocks")

    k_arr = np.asarray(k_per_group, dtype=np.int64)
    if k_arr.ndim == 0:
        k_arr = np.full(num_groups, int(k_arr), dtype=np.int64)
    if k_arr.shape != (num_groups,):
        raise ValueError(
            f"k_per_group must be an int or shape ({num_groups},), got {k_arr.shape}"
        )
    if np.any(k_arr < 0):
        raise ValueError("k_per_group entries must be non-negative")
    if np.any(k_arr > num_blocks):
        raise ValueError(
            f"k_per_group entry exceeds the number of blocks ({num_blocks})"
        )

    pinned = np.asarray(pinned_block_indices, dtype=np.int64)
    if pinned.ndim != 1:
        raise ValueError(
            f"pinned_block_indices must be 1-D, got shape {pinned.shape}"
        )
    if np.any(pinned < 0) or np.any(pinned >= num_blocks):
        raise IndexError(
            f"pinned block index out of range [0, {num_blocks}); got {pinned}"
        )
    if pinned.size != np.unique(pinned).size:
        raise ValueError("pinned_block_indices must be unique")

    pinned_set = set(int(b) for b in pinned)

    # Pin anchor pages: +inf importance so they rank above every finite block.
    scores = scores.copy()
    if pinned.size:
        scores[:, pinned] = _INF

    # Dynamic pool = all blocks except the permanent anchors.
    dynamic_mask = np.ones(num_blocks, dtype=bool)
    if pinned.size:
        dynamic_mask[pinned] = False

    # Deterministic processing order: most focused (lowest entropy) group first.
    # Focus is proxied by the total finite importance over the dynamic pool
    # (peaked attention sums to a smaller total than flat attention).
    if dynamic_mask.any():
        focus = np.where(np.isfinite(scores), scores, 0.0)[:, dynamic_mask].sum(axis=1)
    else:
        focus = np.zeros(num_groups, dtype=np.float64)
    order = np.argsort(focus, kind="stable")

    selected_dynamic: set[int] = set()
    per_group = [np.zeros(0, dtype=np.int64) for _ in range(num_groups)]
    for row in order:
        k = int(k_arr[row])
        available = dynamic_mask.copy()
        for block in selected_dynamic:
            available[block] = False
        # Descending order by score: pinned (+inf) lead but are excluded via
        # `available`; among finite dynamic blocks the highest win, ties broken
        # towards the lowest index (stable sort).
        order_desc = np.argsort(-scores[row], kind="stable")
        chosen = np.asarray(
            [int(b) for b in order_desc if available[b]][:k], dtype=np.int64
        )
        for block in chosen.tolist():
            selected_dynamic.add(int(block))
        per_group[row] = np.sort(chosen)

    result = np.asarray(sorted(pinned_set | selected_dynamic), dtype=np.int64)
    return PagedSelection(
        selected_blocks=result,
        per_group_selected=tuple(per_group),
        num_groups=num_groups,
        k_per_group=k_arr,
        pinned_block_indices=pinned.copy(),
    )


@dataclass
class PagedSelection:
    """Per-group block selection for one layer."""

    selected_blocks: npt.NDArray[np.int64]
    """Sorted, duplicate-free selected block indices (union over groups)."""

    per_group_selected: tuple[npt.NDArray[np.int64], ...]
    """Tuple of length ``G`` with each group's selected block indices."""

    num_groups: int
    """Number of KV groups :math:`H_{\\text{KV}}``."""

    k_per_group: npt.NDArray[np.int64]
    """Block budget :math:`K_l` per group."""

    pinned_block_indices: npt.NDArray[np.int64]
    """Anchor pages pinned with :math:`+\\infty` importance."""


__all__ = [
    "PagedSelection",
    "block_max_reduce",
    "block_sum_reduce",
    "pin_blocks",
    "select_blocks",
    "select_top_k_per_group",
]
