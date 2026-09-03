"""Discrete Renyi-2 layer budget allocation (Scenario 2).

Theory
------
Let :math:`A_{l,h} \\in \\mathbb{R}^{T}` be the attention distribution of logical
query head :math:`h` at layer :math:`l` (typically the *last*-token attention row
over the :math:`T` sequence positions).  The discrete collision (Renyi-2,
:math:`\\alpha = 2`) entropy is

.. math::
    \\mathcal{H}_2(A_{l,h}) =
    -\\ln \\sum_{t=1}^{T} (A_{l,h}(t))^2
    = -\\ln \\Vert A_{l,h} \\Vert_2^2.

Because :math:`A_{l,h}` sums to one, the collision entropy is non-negative,
equals :math:`\\ln T` for a uniform distribution, and equals zero for a one-hot
distribution.  High entropy means *flat* attention (the head needs many tokens);
low entropy means *focused* attention (the head only needs a few tokens).  We
therefore give focused (low-entropy) layers a larger relative budget:

.. math::
    \\bar{\\mathcal{H}}_2^{(l)} =
    \\frac{1}{H_Q} \\sum_{h=1}^{H_Q} \\mathcal{H}_2(A_{l,h}),

.. math::
    \\mathcal{B}_{\\text{dynamic}} =
    \\mathcal{B}_{\\text{total}} - \\lvert \\mathcal{P}_{\\text{pinned}} \\rvert
    \\cdot B,

.. math::
    \\mathcal{B}_l = \\max\\left(1, \\;
    \\left\\lfloor \\mathcal{B}_{\\text{dynamic}} \\cdot
    \\frac{\\exp(-\\bar{\\mathcal{H}}_2^{(l)}/\\tau_L)}
         {\\sum_{j=1}^{L} \\exp(-\\bar{\\mathcal{H}}_2^{(j)}/\\tau_L)}
    \\right\\rfloor \\right).

The per-layer budgets are computed over an observation window
:math:`W_{\\text{obs}}` (EMA smoothing of the layer entropies) so that a single
decode step cannot thrash the allocation.

Numerical stability
-------------------
The normalisation :math:`\\epsilon = 10^{-12}` is applied *inside* the logarithm
only, as a floor on the collision value:

.. math::
    \\mathcal{H}_2 = -\\ln\\left(\\max\\left(\\Vert A \\Vert_2^2, \\epsilon\\right)\\right).

Because :math:`\\sum_t A(t) = 1` is preserved exactly and
:math:`\\Vert A \\Vert_2^2 \\ge 1/T > 0` for any normalised distribution, the
floor only activates for astronomically long sequences (:math:`T > 10^{12}`)
where the exact collision underflows the floating-point range; it therefore never
distorts the entropy of a real distribution while keeping :math:`\\mathcal{H}_2`
non-negative (a one-hot distribution yields exactly ``0`` rather than a tiny
negative value or ``-inf``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

#: Collision-entropy normalisation floor (applied inside the logarithm only).
ENTROPY_EPSILON: float = 1e-12


def collision_entropy(
    attention: npt.ArrayLike, epsilon: float = ENTROPY_EPSILON
) -> float:
    """Discrete collision (Renyi-2) entropy of an attention distribution.

    Parameters
    ----------
    attention:
        Attention distribution :math:`A` of shape ``(T,)`` with non-negative
        entries summing to one.
    epsilon:
        Floor applied to the collision *inside* the logarithm to keep
        ``ln(0)`` finite.  It does *not* renormalise the distribution.

    Returns
    -------
    float
        :math:`-\\ln(\\max(\\Vert A \\Vert_2^2, \\epsilon))`, a non-negative
        scalar.
    """
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(
            f"collision_entropy expects a 1-D distribution, got shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError("attention distribution is empty (zero sequence length)")
    if np.any(a < 0):
        raise ValueError("attention weights must be non-negative")
    if a.sum() <= 0.0:
        raise ValueError("attention distribution has zero total mass")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    collision = float(np.dot(a, a))
    return float(-np.log(max(collision, epsilon)))


def collision_entropies(
    attention: npt.ArrayLike, epsilon: float = ENTROPY_EPSILON
) -> npt.NDArray[np.float64]:
    """Vectorised collision entropy over a 2-D attention tensor.

    Parameters
    ----------
    attention:
        Attention tensor of shape ``(H, T)`` or ``(L, H, T)``.
    epsilon:
        Logarithm floor.

    Returns
    -------
    ndarray
        Shape ``(H,)`` or ``(L, H,)`` with per-head entropies.
    """
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim not in (2, 3):
        raise ValueError(
            f"attention must be 2-D (H, T) or 3-D (L, H, T), got shape {a.shape}"
        )
    if np.any(a < 0):
        raise ValueError("attention weights must be non-negative")
    if a.shape[-1] == 0:
        raise ValueError("attention has zero sequence length")
    if not np.allclose(a.sum(axis=-1), 1.0, atol=1e-4):
        raise ValueError("attention rows must be normalised to sum to one")

    collision = np.einsum("...t,...t->...", a, a)
    return -np.log(np.maximum(collision, epsilon))


def layer_mean_entropies(
    attention: npt.ArrayLike, epsilon: float = ENTROPY_EPSILON
) -> npt.NDArray[np.float64]:
    """Layer-mean collision entropies :math:`\\bar{\\mathcal{H}}_2^{(l)}`.

    Parameters
    ----------
    attention:
        Attention tensor of shape ``(L, H, T)``.

    Returns
    -------
    ndarray
        Shape ``(L,)``.
    """
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(
            f"layer_mean_entropies expects (L, H, T), got shape {a.shape}"
        )
    entropies = collision_entropies(a, epsilon=epsilon)
    return entropies.mean(axis=1)


def softmax_layer_weights(
    layer_entropies: Sequence[float], temperature: float
) -> npt.NDArray[np.float64]:
    """Exponentially-downweighted, normalised layer weights.

    Returns :math:`\\exp(-\\mathcal{H}_2^{(l)}/\\tau_L)` normalised to sum to
    one, using the numerically stable ``max - logsumexp`` trick.  ``temperature``
    must be strictly positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    h = np.asarray(layer_entropies, dtype=np.float64)
    if h.size == 0:
        return h.copy()
    logits = -h / float(temperature)
    logits = logits - logits.max()
    weights = np.exp(logits)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("softmax produced zero total weight")
    return weights / total


def allocate_layer_token_budgets(
    layer_entropies: Sequence[float],
    dynamic_budget: int,
    temperature: float = 1.0,
) -> npt.NDArray[np.int64]:
    """Allocate dynamic *token* budgets per layer.

    .. math::
        \\mathcal{B}_l = \\max\\left(1, \\;
        \\left\\lfloor \\mathcal{B}_{\\text{dynamic}} \\cdot
        \\frac{\\exp(-\\bar{\\mathcal{H}}_2^{(l)}/\\tau_L)}
             {\\sum_j \\exp(-\\bar{\\mathcal{H}}_2^{(j)}/\\tau_L)}
        \\right\\rfloor \\right).

    Returns an integer array of shape ``(L,)`` with every entry :math:`\\ge 1`.
    """
    h = np.asarray(layer_entropies, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"layer_entropies must be 1-D, got shape {h.shape}")
    if dynamic_budget < 0:
        raise ValueError(f"dynamic_budget must be non-negative, got {dynamic_budget}")
    num_layers = h.size
    if num_layers == 0:
        return np.zeros(0, dtype=np.int64)

    weights = softmax_layer_weights(h, temperature)
    budgets = np.floor(dynamic_budget * weights).astype(np.int64)
    budgets = np.maximum(budgets, 1)
    return budgets


def blocks_per_group(
    layer_budgets: Sequence[int], num_kv_heads: int, block_size: int
) -> npt.NDArray[np.int64]:
    """Convert per-layer *token* budgets to per-layer per-KV-group block counts.

    .. math::
        K_l = \\left\\lfloor \\mathcal{B}_l / (H_{\\text{KV}} \\cdot B) \\right\\rfloor.

    The result may be zero: a layer whose token budget is smaller than one
    block per KV group keeps only its pinned anchor pages (no dynamic blocks).
    """
    budgets = np.asarray(layer_budgets, dtype=np.int64)
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if budgets.ndim != 1:
        raise ValueError(f"layer_budgets must be 1-D, got shape {budgets.shape}")

    per_group_budget = budgets // (num_kv_heads * block_size)
    return per_group_budget.astype(np.int64)


@dataclass
class LayerBudget:
    """Per-layer dynamic budget for one observation step."""

    layer_entropies: npt.NDArray[np.float64]
    """Layer-mean entropies, shape ``(L,)``."""

    layer_token_budgets: npt.NDArray[np.int64]
    """Dynamic token budgets :math:`\\mathcal{B}_l`, shape ``(L,)``."""

    layer_block_budgets: npt.NDArray[np.int64]
    """Per-KV-group block counts :math:`K_l`, shape ``(L,)``."""

    dynamic_token_budget: int
    """Total dynamic token budget :math:`\\mathcal{B}_{\\text{dynamic}}``."""

    temperature: float
    """Layer temperature :math:`\\tau_L` used for the allocation."""


class EntropyBudgetAllocator:
    """Smoothing allocator that turns attention tensors into per-layer budgets.

    Layer entropies are tracked with an exponential moving average over an
    observation window :math:`W_{\\text{obs}}` (``smoothing``), which prevents
    decode-step budget thrashing: a single outlying step only moves the smoothed
    estimate by a factor :math:`(1 - \\alpha)`.
    """

    def __init__(self, temperature: float = 1.0, smoothing: float = 0.7) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        self.temperature = temperature
        self.smoothing = smoothing
        self._state: Optional[npt.NDArray[np.float64]] = None
        self.num_observations = 0

    def update(
        self, attention: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Update the smoothed layer-entropy state and return it.

        Parameters
        ----------
        attention:
            Attention tensor of shape ``(L, H, T)``.

        Returns
        -------
        ndarray
            Smoothed layer-mean entropies of shape ``(L,)``.
        """
        current = layer_mean_entropies(attention)
        if self._state is None or self._state.shape != current.shape:
            self._state = current.copy()
        else:
            self._state = (
                self.smoothing * self._state + (1.0 - self.smoothing) * current
            )
        self.num_observations += 1
        return self._state.copy()

    def allocate(
        self,
        dynamic_budget: int,
        num_kv_heads: int,
        block_size: int,
    ) -> LayerBudget:
        """Allocate budgets from the current smoothed entropy state.

        Returns a :class:`LayerBudget` with per-layer token budgets and
        per-KV-group block counts.  If no attention has been observed yet the
        allocation is uniform.
        """
        if dynamic_budget < 0:
            raise ValueError(
                f"dynamic_budget must be non-negative, got {dynamic_budget}"
            )
        if self._state is None:
            raise RuntimeError(
                "allocate() called before update(); call update() with an "
                "attention tensor first"
            )
        token_budgets = allocate_layer_token_budgets(
            self._state, dynamic_budget, temperature=self.temperature
        )
        block_budgets = blocks_per_group(token_budgets, num_kv_heads, block_size)
        return LayerBudget(
            layer_entropies=self._state.copy(),
            layer_token_budgets=token_budgets,
            layer_block_budgets=block_budgets,
            dynamic_token_budget=int(dynamic_budget),
            temperature=self.temperature,
        )

    def reset(self) -> None:
        """Forget the smoothed entropy state."""
        self._state = None
        self.num_observations = 0


__all__ = [
    "ENTROPY_EPSILON",
    "EntropyBudgetAllocator",
    "LayerBudget",
    "allocate_layer_token_budgets",
    "blocks_per_group",
    "collision_entropies",
    "collision_entropy",
    "layer_mean_entropies",
    "softmax_layer_weights",
]
