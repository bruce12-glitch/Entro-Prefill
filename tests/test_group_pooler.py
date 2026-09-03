"""Tests for Scenario 3 (first half): GQA group pooling."""

from __future__ import annotations

import numpy as np
import pytest

from page_entrokv.group_pooler import (
    GroupPooler,
    group_index,
    group_members,
    intra_group_weights,
    pool_attention,
)


def test_group_index_mapping() -> None:
    # H_Q = 8, H_KV = 2 -> ratio 4.
    assert group_index(0, num_kv_heads=2, num_query_heads=8) == 0
    assert group_index(3, num_kv_heads=2, num_query_heads=8) == 0
    assert group_index(4, num_kv_heads=2, num_query_heads=8) == 1
    assert group_index(7, num_kv_heads=2, num_query_heads=8) == 1


def test_group_index_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError):
        group_index(0, num_kv_heads=3, num_query_heads=8)
    with pytest.raises(IndexError):
        group_index(8, num_kv_heads=2, num_query_heads=8)


def test_group_members() -> None:
    members = group_members(0, num_kv_heads=2, num_query_heads=8)
    assert members.tolist() == [0, 1, 2, 3]
    members = group_members(1, num_kv_heads=2, num_query_heads=8)
    assert members.tolist() == [4, 5, 6, 7]


def test_intra_group_weights_normalise_per_group() -> None:
    entropies = np.array([0.1, 0.5, 2.0, 2.5])
    w0 = intra_group_weights(entropies, 0, 1.0, num_kv_heads=2, num_query_heads=4)
    w1 = intra_group_weights(entropies, 1, 1.0, num_kv_heads=2, num_query_heads=4)
    assert w0.sum() == pytest.approx(1.0)
    assert w1.sum() == pytest.approx(1.0)
    # Focused heads receive larger weights within their group.
    assert w0[0] > w0[1]
    assert w1[0] > w1[1]


def test_pooler_pool_attention_shapes_and_normalisation() -> None:
    rng = np.random.default_rng(4)
    attention = rng.random((8, 32))
    attention /= attention.sum(axis=-1, keepdims=True)
    pooler = GroupPooler(num_kv_heads=2, num_query_heads=8, temperature=1.0)
    pooled, weights = pooler.pool_attention(attention)
    assert pooled.shape == (2, 32)
    assert weights.shape == (8,)
    assert np.allclose(pooled.sum(axis=-1), 1.0, atol=1e-9)
    # Pooled distribution is a convex combination of member distributions.
    members = pooler.group_members(0)
    expected = np.tensordot(weights[members], attention[members], axes=(0, 0))
    assert np.allclose(pooled[0], expected)


def test_pooler_rejects_mismatched_heads() -> None:
    pooler = GroupPooler(num_kv_heads=2, num_query_heads=8)
    bad = np.ones((4, 16)) / 16
    with pytest.raises(ValueError):
        pooler.pool_attention(bad)


def test_pooler_rejects_unnormalised_attention() -> None:
    pooler = GroupPooler(num_kv_heads=2, num_query_heads=8)
    bad = np.ones((8, 16))  # rows sum to 16, not 1
    with pytest.raises(ValueError):
        pooler.pool_attention(bad)


def test_functional_pool_attention_matches_class() -> None:
    rng = np.random.default_rng(5)
    attention = rng.random((8, 16))
    attention /= attention.sum(axis=-1, keepdims=True)
    pooled_fn, weights_fn = pool_attention(attention, 2, 8, temperature=1.0)
    pooler = GroupPooler(num_kv_heads=2, num_query_heads=8, temperature=1.0)
    pooled_cls, weights_cls = pooler.pool_attention(attention)
    assert np.allclose(pooled_fn, pooled_cls)
    assert np.allclose(weights_fn, weights_cls)
