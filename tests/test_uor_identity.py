"""Property tests for the Unique Occupancy Ratio (UOR) identity."""

from __future__ import annotations

import numpy as np
import pytest

from page_entrokv.metrics import (
    compaction,
    coverage,
    eviction_recall,
    occupancy_counts,
    retention_ratio,
    token_retention,
    unique_occupancy_ratio,
)
from page_entrokv.engine import PageEntroKVConfig, PageEntroKVEngine
from page_entrokv.paged_mapping import select_top_k_per_group


def test_uor_is_exactly_one_across_many_configurations() -> None:
    """Fuzz the selection with many geometries and assert UOR == 1.00 exactly."""
    rng = np.random.default_rng(20240903)
    for trial in range(200):
        num_groups = int(rng.integers(1, 6))
        num_blocks = int(rng.integers(4, 64))
        k = int(rng.integers(1, num_blocks + 1))
        scores = rng.random((num_groups, num_blocks))
        num_pinned = int(rng.integers(0, min(4, num_blocks) + 1))
        pinned = np.sort(
            rng.choice(num_blocks, size=num_pinned, replace=False).astype(np.int64)
        )
        selection = select_top_k_per_group(
            scores, k_per_group=k, pinned_block_indices=pinned
        )
        # Pinned pages must always survive.
        assert set(pinned.tolist()).issubset(set(selection.selected_blocks.tolist()))
        # The per-group selections must be pairwise disjoint (UOR == 1.0 on the
        # dynamic selection), exactly, not approximately.
        uor = unique_occupancy_ratio(selection.per_group_selected, num_blocks)
        assert uor == 1.0, f"UOR violated at trial {trial}"
        # The flat union must equal the pinned pages plus the disjoint dynamic
        # selections.
        expected = set(pinned.tolist())
        for group_blocks in selection.per_group_selected:
            expected.update(int(b) for b in group_blocks)
        assert set(selection.selected_blocks.tolist()) == expected


def test_uor_one_hot_like_focus_survives() -> None:
    """A single extremely peaked group still keeps its block under contention."""
    scores = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    selection = select_top_k_per_group(scores, k_per_group=1, pinned_block_indices=[])
    assert 0 in selection.selected_blocks
    assert unique_occupancy_ratio(selection.per_group_selected, 5) == 1.0


def test_uor_definition_for_duplicate_selection_is_detected() -> None:
    # A deliberately duplicated selection must be detected as UOR < 1.
    selected = [np.array([1, 2]), np.array([1, 3])]
    uor = unique_occupancy_ratio(selected, num_blocks=4)
    assert uor == pytest.approx(2 / 3)


def test_uor_empty_selection_is_vacuous_one() -> None:
    # "Every occupied block is uniquely occupied" is vacuously true when nothing
    # is occupied, so the UOR is 1.0 for an empty selection.
    assert unique_occupancy_ratio([], num_blocks=8) == 1.0


def test_occupancy_counts() -> None:
    counts = occupancy_counts([np.array([0, 1]), np.array([1, 2])], num_blocks=4)
    assert counts.tolist() == [1, 2, 1, 0]


def test_occupancy_counts_flat() -> None:
    counts = occupancy_counts(np.array([0, 1, 1]), num_blocks=3)
    assert counts.tolist() == [1, 2, 0]


def test_coverage_and_compaction() -> None:
    selected = [np.array([0, 1]), np.array([1, 2])]
    assert coverage(selected, num_blocks=4) == pytest.approx(3 / 4)
    assert compaction(selected, num_blocks=4) == pytest.approx(4 / 3)


def test_eviction_recall_and_retention() -> None:
    selected = [0, 1, 2]
    oracle = [0, 2, 5]
    assert eviction_recall(selected, oracle) == pytest.approx(2 / 3)
    assert retention_ratio([0, 2], selected) == pytest.approx(1.0)
    assert retention_ratio([], selected) == pytest.approx(1.0)
    assert retention_ratio([0, 9], selected) == pytest.approx(0.5)


def _attention(num_layers: int, num_heads: int, num_tokens: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    attention = rng.random((num_layers, num_heads, num_tokens))
    attention /= attention.sum(axis=-1, keepdims=True)
    return attention


CODE_PROMPT = (
    "import os\n"
    "from typing import List\n"
    "GLOBAL_LIMIT = 128\n\n"
    "def lookup(key: str) -> List[int]:\n"
    "    return [ord(c) for c in key]\n\n"
    "class Store:\n"
    "    items: List[int] = []\n"
)


def test_engine_end_to_end_preserves_uor_and_pins() -> None:
    config = PageEntroKVConfig(
        block_size=4, pin_ratio=0.30, num_kv_heads=2, num_query_heads=4
    )
    engine = PageEntroKVEngine(config)
    num_tokens = engine.slicer.token_count(CODE_PROMPT)
    num_layers = 4
    attention = _attention(num_layers, config.num_query_heads, num_tokens, seed=0)

    total_blocks = int(np.ceil(num_tokens / config.block_size))
    plan = engine.step(CODE_PROMPT, attention, total_blocks=total_blocks)

    assert plan.num_blocks == total_blocks
    assert plan.total_blocks == total_blocks
    assert plan.selected_blocks  # at least pinned pages survive
    pinned = set(plan.pinned_block_indices.tolist())

    for layer_blocks in plan.selected_blocks:
        layer_blocks = np.asarray(layer_blocks)
        # Every layer keeps the pinned anchors.
        assert pinned.issubset(set(layer_blocks.tolist()))
        # Per-layer selection is duplicate-free (UOR == 1.00).
        assert np.unique(layer_blocks).size == layer_blocks.size
        # All selected blocks are within the physical capacity.
        assert layer_blocks.max() < plan.num_blocks


def test_engine_retention_budget_bounds_residency() -> None:
    config = PageEntroKVConfig(
        block_size=4, pin_ratio=0.30, num_kv_heads=2, num_query_heads=4
    )
    engine = PageEntroKVEngine(config)
    num_tokens = engine.slicer.token_count(CODE_PROMPT)
    attention = _attention(6, config.num_query_heads, num_tokens, seed=1)

    physical = int(np.ceil(num_tokens / config.block_size))
    budget = physical // 2
    plan = engine.step(CODE_PROMPT, attention, total_blocks=budget)

    union = (
        np.unique(np.concatenate(plan.selected_blocks))
        if plan.selected_blocks
        else np.array([], dtype=np.int64)
    )
    # Resident pages never exceed the retention budget.
    assert union.size <= plan.total_blocks
    # The budget is below physical capacity, so residency is a strict subset.
    assert union.size < physical


def test_engine_natural_language_degrades_to_dynamic_only() -> None:
    config = PageEntroKVConfig(
        block_size=4, pin_ratio=0.30, num_kv_heads=2, num_query_heads=4
    )
    engine = PageEntroKVEngine(config)
    prose = "The quick brown fox jumps over the lazy dog. " * 8
    num_tokens = engine.slicer.token_count(prose)
    attention = _attention(4, config.num_query_heads, num_tokens, seed=2)

    plan = engine.step(prose, attention, total_blocks=int(np.ceil(num_tokens / 4)))
    # Natural language has no AST invariants -> empty pinned mask.
    assert plan.page_mask.degraded
    assert plan.pinned_block_indices.size == 0
    # The whole dynamic budget goes to entropy-selected blocks; every layer's
    # selection is still duplicate-free.
    for layer_blocks in plan.selected_blocks:
        assert np.unique(layer_blocks).size == layer_blocks.size


def test_token_retention() -> None:
    mask = np.array([True, True, False, False, True, False], dtype=bool)
    # block size 2 -> blocks [T,T]=pinned, [F,F]=no, [T,F]=pinned.
    assert token_retention(mask, block_size=2) == pytest.approx(1.0)
    mask2 = np.array([True, False, True, False], dtype=bool)
    # blocks [T,F] and [T,F] -> both pinned -> all invariant tokens retained.
    assert token_retention(mask2, block_size=2) == pytest.approx(1.0)
    assert token_retention(np.zeros(4, dtype=bool), block_size=2) == pytest.approx(1.0)
