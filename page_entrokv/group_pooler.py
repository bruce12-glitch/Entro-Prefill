"""GQA group pooling for Page-EntroKV (Scenario 3, first half).

Theory
------
Grouped-Query Attention shares :math:`H_{\\text{KV}}` physical KV heads across
:math:`H_Q` logical query heads with group ratio

.. math::
    r = H_Q / H_{\\text{KV}}, \\qquad r \\in \\mathbb{Z}_{> 0}.

Logical query head :math:`h` maps to physical KV head

.. math::
    g = \\lfloor h / r \\rfloor,

and the query heads in group :math:`g` are

.. math::
    \\text{Group}(g) = \\{ h \\mid \\lfloor h / r \\rfloor = g \\}.

Each physical KV head :math:`g` therefore receives one *pooled* attention
distribution, an entropy-weighted mixture of its member query heads using
intra-group Boltzmann weights

.. math::
    w_{l,h} = \\frac{\\exp(-\\mathcal{H}_2(A_{l,h})/\\tau_g)}
                  {\\sum_{j \\in \\text{Group}(g)}
                       \\exp(-\\mathcal{H}_2(A_{l,j})/\\tau_g)},

.. math::
    A_{l,g}^{\\text{pooled}}(t) = \\sum_{h \\in \\text{Group}(g)}
        w_{l,h}\\, A_{l,h}(t).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from page_entrokv.entropy_budget import collision_entropies


def group_index(
    query_head: int, num_kv_heads: int, num_query_heads: int
) -> int:
    """Return the physical KV-head index :math:`g = \\lfloor h / r \\rfloor`."""
    if num_query_heads <= 0:
        raise ValueError(f"num_query_heads must be positive, got {num_query_heads}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be an integer multiple of num_kv_heads "
            f"(got {num_query_heads} / {num_kv_heads})"
        )
    if query_head < 0 or query_head >= num_query_heads:
        raise IndexError(
            f"query_head {query_head} out of range [0, {num_query_heads})"
        )
    ratio = num_query_heads // num_kv_heads
    return query_head // ratio


def group_members(
    group: int, num_kv_heads: int, num_query_heads: int
) -> npt.NDArray[np.int64]:
    """Return :math:`\\text{Group}(g)`, the query heads mapped to KV head ``g``."""
    if num_query_heads <= 0:
        raise ValueError(f"num_query_heads must be positive, got {num_query_heads}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be an integer multiple of num_kv_heads "
            f"(got {num_query_heads} / {num_kv_heads})"
        )
    if group < 0 or group >= num_kv_heads:
        raise IndexError(f"group {group} out of range [0, {num_kv_heads})")
    ratio = num_query_heads // num_kv_heads
    return np.arange(group * ratio, (group + 1) * ratio, dtype=np.int64)


def intra_group_weights(
    entropies: Sequence[float],
    group: int,
    temperature: float,
    num_kv_heads: int,
    num_query_heads: int,
) -> npt.NDArray[np.float64]:
    """Boltzmann weights of query heads within one GQA group.

    Parameters
    ----------
    entropies:
        Per-query-head entropies of shape ``(H_Q,)``.
    group:
        Physical KV head index :math:`g`.
    temperature:
        Group temperature :math:`\\tau_g` (must be positive).
    num_kv_heads:
        Number of physical KV heads :math:`H_{\\text{KV}}`.
    num_query_heads:
        Number of logical query heads :math:`H_Q` (integer multiple of
        ``num_kv_heads``).

    Returns
    -------
    ndarray
        Weight vector of shape ``(r,)`` summing to one, aligned with
        ``group_members(group, num_kv_heads, num_query_heads)``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if num_query_heads <= 0:
        raise ValueError(f"num_query_heads must be positive, got {num_query_heads}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be an integer multiple of num_kv_heads "
            f"(got {num_query_heads} / {num_kv_heads})"
        )
    h = np.asarray(entropies, dtype=np.float64)
    if h.shape != (num_query_heads,):
        raise ValueError(
            f"entropies must have shape ({num_query_heads},), got {h.shape}"
        )
    if group < 0 or group >= num_kv_heads:
        raise IndexError(f"group {group} out of range [0, {num_kv_heads})")

    members = group_members(group, num_kv_heads, num_query_heads)
    logits = -h[members] / float(temperature)
    logits = logits - logits.max()
    expo = np.exp(logits)
    return expo / expo.sum()


class GroupPooler:
    """Pool per-query-head attention into per-KV-head distributions.

    Parameters
    ----------
    num_kv_heads:
        Number of physical KV heads :math:`H_{\\text{KV}}`.
    num_query_heads:
        Number of logical query heads :math:`H_Q` (an integer multiple of
        ``num_kv_heads``).
    temperature:
        Intra-group temperature :math:`\\tau_g`.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_query_heads: int,
        temperature: float = 1.0,
    ) -> None:
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if num_query_heads <= 0:
            raise ValueError(f"num_query_heads must be positive, got {num_query_heads}")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be an integer multiple of num_kv_heads "
                f"(got {num_query_heads} / {num_kv_heads})"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.ratio = num_query_heads // num_kv_heads
        self.temperature = temperature

    def group_members(self, group: int) -> npt.NDArray[np.int64]:
        """Query-head indices mapped to physical KV head ``group``."""
        return group_members(group, self.num_kv_heads, self.num_query_heads)

    def intra_group_weights(self, entropies: Sequence[float]) -> npt.NDArray[np.float64]:
        """All intra-group Boltzmann weights, shape ``(H_Q,)``.

        Weights are computed within each group (softmax over its :math:`r`
        members) so each group's weights sum to one.
        """
        h = np.asarray(entropies, dtype=np.float64)
        if h.shape != (self.num_query_heads,):
            raise ValueError(
                f"entropies must have shape ({self.num_query_heads},), "
                f"got {h.shape}"
            )
        logits = -h / self.temperature
        weights = np.empty_like(h)
        for g in range(self.num_kv_heads):
            members = self.group_members(g)
            subgroup = logits[members]
            subgroup = subgroup - subgroup.max()
            expo = np.exp(subgroup)
            weights[members] = expo / expo.sum()
        return weights

    def pool_attention(
        self, attention: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Pool query-head attention into KV-head distributions.

        Parameters
        ----------
        attention:
            Attention tensor of shape ``(H_Q, T)``.

        Returns
        -------
        pooled:
            Pooled distribution of shape ``(H_KV, T)``.
        weights:
            Intra-group weights of shape ``(H_Q,)`` used for pooling.
        """
        a = np.asarray(attention, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"attention must be (H_Q, T), got shape {a.shape}")
        if a.shape[0] != self.num_query_heads:
            raise ValueError(
                f"attention has {a.shape[0]} heads, expected {self.num_query_heads}"
            )
        if not np.allclose(a.sum(axis=-1), 1.0, atol=1e-4):
            raise ValueError("attention rows must be normalised to sum to one")

        entropies = collision_entropies(a)
        weights = self.intra_group_weights(entropies)
        pooled = np.zeros((self.num_kv_heads, a.shape[1]), dtype=np.float64)
        for g in range(self.num_kv_heads):
            members = self.group_members(g)
            pooled[g] = np.tensordot(weights[members], a[members], axes=(0, 0))
        return pooled, weights


def pool_attention(
    attention: npt.ArrayLike,
    num_kv_heads: int,
    num_query_heads: int,
    temperature: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Functional convenience wrapper around :class:`GroupPooler`.

    Parameters
    ----------
    attention:
        Attention tensor of shape ``(H_Q, T)``.
    num_kv_heads, num_query_heads, temperature:
        See :class:`GroupPooler`.

    Returns
    -------
    pooled:
        Pooled distribution of shape ``(H_KV, T)``.
    weights:
        Intra-group weights of shape ``(H_Q,)`` used for pooling.
    """
    return GroupPooler(num_kv_heads, num_query_heads, temperature).pool_attention(
        attention
    )


__all__ = [
    "GroupPooler",
    "group_index",
    "group_members",
    "intra_group_weights",
    "pool_attention",
]
