"""Tests for Scenario 3 (second half): paged-frame mapping and selection."""

from __future__ import annotations

import numpy as np
import pytest

from page_entrokv.paged_mapping import (
    block_max_reduce,
    block_sum_reduce,
    pin_blocks,
    select_blocks,
    select_top_k_per_group,
)


def test_block_max_reduce_1d() -> None:
    a = np.array([0.1, 0.2, 0.05, 0.3, 0.0, 0.4, 0.1, 0.15, 0.25])
    r = block_max_reduce(a, block_size=3)
    assert r.shape == (3,)
    assert r.tolist() == pytest.approx([0.2, 0.4, 0.25])


def test_block_max_reduce_2d() -> None:
    a = np.array([[0.1, 0.2, 0.05, 0.3], [0.0, 0.1, 0.9, 0.0]])
    r = block_max_reduce(a, block_size=2)
    assert r.shape == (2, 2)
    assert np.allclose(r, [[0.2, 0.3], [0.1, 0.9]])


def test_block_sum_reduce() -> None:
    a = np.array([0.1, 0.2, 0.05, 0.3])
    r = block_sum_reduce(a, block_size=2)
    assert r.tolist() == pytest.approx([0.3, 0.35])


def test_pin_blocks_sets_infinity() -> None:
    scores = np.zeros((2, 5), dtype=np.float64)
    pinned = pin_blocks(scores, [1, 3])
    assert np.isinf(pinned[:, 1]).all()
    assert np.isinf(pinned[:, 3]).all()
    assert np.isfinite(pinned[:, 0]).all()


def test_pin_blocks_does_not_mutate_input() -> None:
    scores = np.zeros(4, dtype=np.float64)
    _ = pin_blocks(scores, [0])
    assert np.isfinite(scores).all()


def test_pin_blocks_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        pin_blocks(np.zeros(3), [5])


def test_select_blocks_top_k() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    assert select_blocks(scores, 2).tolist() == [1, 4]
    assert select_blocks(scores, 0).size == 0
    assert select_blocks(scores, 5).tolist() == [0, 1, 2, 3, 4]


def test_select_blocks_pinned_first() -> None:
    scores = np.array([0.1, np.inf, 0.5, np.inf, 0.8])
    assert select_blocks(scores, 2).tolist() == [1, 3]


def test_select_blocks_rejects_k_too_large() -> None:
    with pytest.raises(ValueError):
        select_blocks(np.ones(3), 4)


def test_select_top_k_per_group_pinned_always_selected() -> None:
    rng = np.random.default_rng(6)
    scores = rng.random((3, 20))
    pinned = np.array([2, 7])
    selection = select_top_k_per_group(scores, k_per_group=3, pinned_block_indices=pinned)
    selected = selection.selected_blocks
    assert set(pinned.tolist()).issubset(set(selected.tolist()))
    assert selected.ndim == 1
    assert np.array_equal(selected, np.sort(selected))


def test_select_top_k_per_group_no_duplicates() -> None:
    rng = np.random.default_rng(7)
    scores = rng.random((4, 30))
    selection = select_top_k_per_group(scores, k_per_group=5, pinned_block_indices=[])
    assert np.unique(selection.selected_blocks).size == selection.selected_blocks.size


def test_select_top_k_per_group_per_group_budget() -> None:
    # Uniform importance -> each group selects the lowest-index k available
    # blocks; duplicates are forbidden so groups take distinct blocks, giving
    # k * num_groups selected in total when the pool is large enough.
    scores = np.full((3, 20), 0.5)
    selection = select_top_k_per_group(scores, k_per_group=4, pinned_block_indices=[])
    selected = selection.selected_blocks
    assert selected.size == 12
    assert np.unique(selected).size == 12
    # Ties break towards the lowest indices: groups grab 0..11.
    assert selected.tolist() == list(range(12))


def test_select_top_k_per_group_pinned_do_not_consume_budget() -> None:
    # Anchor pages are permanent: they are returned even when k=0 and are never
    # double-counted against the dynamic budget.
    scores = np.random.default_rng(9).random((2, 8))
    selection = select_top_k_per_group(scores, k_per_group=0, pinned_block_indices=[3, 5])
    assert selection.selected_blocks.tolist() == [3, 5]
    # With a positive budget the pinned pages survive and the dynamic blocks are
    # distinct from them.
    selection = select_top_k_per_group(scores, k_per_group=2, pinned_block_indices=[3, 5])
    assert 3 in selection.selected_blocks and 5 in selection.selected_blocks
    assert np.unique(selection.selected_blocks).size == selection.selected_blocks.size


def test_select_top_k_per_group_zero_budget() -> None:
    scores = np.random.default_rng(8).random((2, 5))
    selection = select_top_k_per_group(scores, k_per_group=0, pinned_block_indices=[])
    assert selection.selected_blocks.size == 0


def test_select_top_k_per_group_validates_inputs() -> None:
    scores = np.ones((2, 4))
    with pytest.raises(ValueError):
        select_top_k_per_group(scores, k_per_group=[1, 2, 3], pinned_block_indices=[])
    with pytest.raises(ValueError):
        select_top_k_per_group(scores, k_per_group=5, pinned_block_indices=[])
    with pytest.raises(IndexError):
        select_top_k_per_group(scores, k_per_group=1, pinned_block_indices=[9])
    with pytest.raises(ValueError):
        select_top_k_per_group(scores, k_per_group=1, pinned_block_indices=[0, 0])
