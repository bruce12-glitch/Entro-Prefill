"""Tests for Scenario 2: discrete Renyi-2 (collision) entropy and budgets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from page_entrokv.entropy_budget import (
    EntropyBudgetAllocator,
    allocate_layer_token_budgets,
    blocks_per_group,
    collision_entropies,
    collision_entropy,
    layer_mean_entropies,
    softmax_layer_weights,
)


def test_collision_entropy_one_hot_is_zero() -> None:
    a = np.zeros(16)
    a[3] = 1.0
    h = collision_entropy(a)
    # -ln(1 + 1e-12) ~ 0.
    assert h == pytest.approx(0.0, abs=1e-9)


def test_collision_entropy_uniform_is_ln_t() -> None:
    t = 8
    a = np.full(t, 1.0 / t)
    assert collision_entropy(a) == pytest.approx(math.log(t), abs=1e-9)


def test_collision_entropy_is_nonnegative_and_bounded() -> None:
    rng = np.random.default_rng(0)
    for t in (1, 3, 16, 64):
        a = rng.random(t)
        a /= a.sum()
        h = collision_entropy(a)
        assert h >= 0.0
        assert h <= math.log(t) + 1e-9


def test_collision_entropy_preserves_normalisation() -> None:
    # The epsilon must live inside the log only; the distribution is unchanged.
    a = np.array([1.0, 0.0, 0.0])
    assert a.sum() == 1.0
    h = collision_entropy(a)
    assert np.isfinite(h)
    assert h >= 0.0


def test_collision_entropy_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        collision_entropy(np.zeros(0))
    with pytest.raises(ValueError):
        collision_entropy(np.array([1.0, -0.5, 0.5]))
    with pytest.raises(ValueError):
        collision_entropy(np.zeros(3))
    with pytest.raises(ValueError):
        collision_entropy(np.ones((2, 2)))


def test_collision_entropies_vectorised() -> None:
    rng = np.random.default_rng(1)
    a = rng.random((5, 32))
    a /= a.sum(axis=-1, keepdims=True)
    hs = collision_entropies(a)
    assert hs.shape == (5,)
    for i in range(5):
        assert hs[i] == pytest.approx(collision_entropy(a[i]))


def test_layer_mean_entropies() -> None:
    rng = np.random.default_rng(2)
    a = rng.random((4, 8, 64))
    a /= a.sum(axis=-1, keepdims=True)
    means = layer_mean_entropies(a)
    assert means.shape == (4,)
    assert np.allclose(means, collision_entropies(a).mean(axis=1))


def test_softmax_layer_weights_normalise() -> None:
    h = np.array([0.5, 1.0, 1.5, 2.0])
    w = softmax_layer_weights(h, temperature=1.0)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0)
    # Lower entropy -> higher weight.
    assert w[0] > w[-1]


def test_softmax_layer_weights_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError):
        softmax_layer_weights([1.0, 2.0], temperature=0.0)


def test_allocate_layer_token_budgets_floor_and_focus() -> None:
    h = np.array([0.1, 2.0])  # layer 0 focused, layer 1 flat
    budgets = allocate_layer_token_budgets(h, dynamic_budget=1000, temperature=1.0)
    assert budgets.shape == (2,)
    assert (budgets >= 1).all()
    assert budgets[0] > budgets[1]
    assert budgets.sum() <= 1000


def test_allocate_layer_token_budgets_uniform_for_equal_entropy() -> None:
    h = np.full(4, 1.5)
    budgets = allocate_layer_token_budgets(h, dynamic_budget=400, temperature=1.0)
    assert budgets.min() == budgets.max()


def test_allocate_layer_token_budgets_small_budget_floor() -> None:
    h = np.array([0.0, 2.0, 2.0])
    budgets = allocate_layer_token_budgets(h, dynamic_budget=2, temperature=1.0)
    assert budgets.sum() >= 2  # flooring guarantees at least 1 token per layer.


def test_blocks_per_group_formula() -> None:
    # K_l = floor(B_l / (H_KV * B)): 512/64 = 8, 511/64 = 7, 63/64 = 0.
    k = blocks_per_group([512, 511, 63], num_kv_heads=4, block_size=16)
    assert k.tolist() == [8, 7, 0]


def test_blocks_per_group_zero_budget_is_zero() -> None:
    k = blocks_per_group([0], num_kv_heads=8, block_size=16)
    assert k.tolist() == [0]


def test_allocator_smooths_and_allocates() -> None:
    rng = np.random.default_rng(3)
    alloc = EntropyBudgetAllocator(temperature=1.0, smoothing=0.9)
    a = rng.random((3, 8, 32))
    a /= a.sum(axis=-1, keepdims=True)
    state1 = alloc.update(a)
    assert state1.shape == (3,)
    state2 = alloc.update(a * 0.0 + (1.0 / 32))
    # Heavily smoothed: the state should barely move after one flat step.
    assert np.allclose(state1, state2, atol=0.2)
    plan = alloc.allocate(dynamic_budget=1024, num_kv_heads=4, block_size=16)
    assert plan.layer_token_budgets.shape == (3,)
    assert plan.layer_block_budgets.shape == (3,)
    assert (plan.layer_token_budgets >= 1).all()


def test_allocator_requires_update_before_allocate() -> None:
    alloc = EntropyBudgetAllocator()
    with pytest.raises(RuntimeError):
        alloc.allocate(dynamic_budget=10, num_kv_heads=2, block_size=16)
