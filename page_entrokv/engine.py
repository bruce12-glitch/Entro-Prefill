"""Page-EntroKV engine: end-to-end orchestration of the three scenarios.

The engine ties the pieces together for one observation step:

1. **AST slicing** (``ASTSlicer``) produces the permanent page mask
   :math:`\\mathcal{M}_{\\text{AST}}` over :math:`\\mathcal{B}_{\\text{total}}`
   pages of size :math:`B`.
2. **Renyi-2 budgets** (``EntropyBudgetAllocator``) convert the observed
   attention tensor :math:`A \\in \\mathbb{R}^{L \\times H_Q \\times T}` into
   per-layer dynamic token budgets :math:`\\mathcal{B}_l`, where
   :math:`\\mathcal{B}_{\\text{dynamic}} =
   \\mathcal{B}_{\\text{total}} - |\\mathcal{P}_{\\text{pinned}}| \\cdot B`.
3. **Group pooling + paged mapping** (``GroupPooler`` +
   ``select_top_k_per_group``) produce, for every layer, the physical pages each
   KV group keeps — pinning anchors at :math:`+\\infty` and enforcing
   :math:`\\text{UOR} \\equiv 1.00`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from page_entrokv.ast_slicing import ASTSlicer, PageMask
from page_entrokv.entropy_budget import EntropyBudgetAllocator
from page_entrokv.group_pooler import GroupPooler
from page_entrokv.paged_mapping import (
    block_max_reduce,
    pin_blocks,
    select_top_k_per_group,
)


@dataclass
class PageEntroKVConfig:
    """Hyperparameters of the Page-EntroKV engine."""

    block_size: int = 16
    """Tokens per physical page :math:`B`."""

    pin_ratio: float = 0.30
    """Cap ratio :math:`\\gamma` bounding the pinned-page budget."""

    num_kv_heads: int = 8
    """Number of physical KV heads :math:`H_{\\text{KV}}``."""

    num_query_heads: int = 32
    """Number of logical query heads :math:`H_Q` (multiple of ``num_kv_heads``)."""

    layer_temperature: float = 1.0
    """Layer-allocation temperature :math:`\\tau_L``."""

    group_temperature: float = 1.0
    """Intra-group Boltzmann temperature :math:`\\tau_g``."""

    smoothing: float = 0.7
    """EMA factor over the observation window :math:`W_{\\text{obs}}``."""

    language: str = "python"
    """tree-sitter language used for AST slicing."""

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.pin_ratio <= 1.0:
            raise ValueError(f"pin_ratio must be in [0, 1], got {self.pin_ratio}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.num_query_heads <= 0:
            raise ValueError(f"num_query_heads must be positive, got {self.num_query_heads}")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be an integer multiple of num_kv_heads "
                f"(got {self.num_query_heads} / {self.num_kv_heads})"
            )
        if self.layer_temperature <= 0.0:
            raise ValueError(f"layer_temperature must be positive, got {self.layer_temperature}")
        if self.group_temperature <= 0.0:
            raise ValueError(f"group_temperature must be positive, got {self.group_temperature}")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {self.smoothing}")


@dataclass
class EvictionPlan:
    """Complete page-eviction plan for one observation step."""

    page_mask: PageMask
    """AST-aware permanent page mask (Scenario 1)."""

    layer_entropies: npt.NDArray[np.float64]
    """Smoothed layer-mean entropies, shape ``(L,)``."""

    layer_token_budgets: npt.NDArray[np.int64]
    """Per-layer dynamic token budgets :math:`\\mathcal{B}_l`, shape ``(L,)``."""

    layer_block_budgets: npt.NDArray[np.int64]
    """Per-layer per-KV-group block counts :math:`K_l`, shape ``(L,)``."""

    selected_blocks: tuple[npt.NDArray[np.int64], ...]
    """Per-layer selected (kept) block indices, each sorted and duplicate-free."""

    pinned_block_indices: npt.NDArray[np.int64]
    """Anchor pages :math:`\\mathcal{P}_{\\text{pinned}}``."""

    total_blocks: int
    """Retention budget :math:`\\mathcal{B}_{\\text{total}}` (blocks to keep)."""

    num_blocks: int
    """Physical capacity :math:`N = \\lceil T / B \\rceil` (blocks in the cache)."""

    dynamic_token_budget: int
    """Dynamic token budget :math:`\\mathcal{B}_{\\text{dynamic}}``."""


class PageEntroKVEngine:
    """Entropy-guided, AST-aware KV-page selector for GQA paged attention.

    Parameters
    ----------
    config:
        Engine hyperparameters.
    tokenizer:
        Optional tokenizer callable (see :class:`ASTSlicer`).
    parser:
        Optional pre-built tree-sitter parser.
    """

    def __init__(
        self,
        config: Optional[PageEntroKVConfig] = None,
        tokenizer=None,
        parser=None,
    ) -> None:
        self.config = config if config is not None else PageEntroKVConfig()
        self.slicer = ASTSlicer(
            language=self.config.language,
            block_size=self.config.block_size,
            pin_ratio=self.config.pin_ratio,
            tokenizer=tokenizer,
            parser=parser,
        )
        self.allocator = EntropyBudgetAllocator(
            temperature=self.config.layer_temperature,
            smoothing=self.config.smoothing,
        )
        self.pooler = GroupPooler(
            num_kv_heads=self.config.num_kv_heads,
            num_query_heads=self.config.num_query_heads,
            temperature=self.config.group_temperature,
        )

    @property
    def parser_available(self) -> bool:
        """``True`` iff AST slicing is active (a parser is available)."""
        return self.slicer.parser_available

    def _validate_attention(
        self, attention: npt.ArrayLike, num_tokens: int
    ) -> npt.NDArray[np.float64]:
        a = np.asarray(attention, dtype=np.float64)
        if a.ndim != 3:
            raise ValueError(
                f"attention must be (L, H_Q, T), got shape {a.shape}"
            )
        if a.shape[1] != self.config.num_query_heads:
            raise ValueError(
                f"attention has {a.shape[1]} query heads, expected "
                f"{self.config.num_query_heads}"
            )
        if a.shape[2] != num_tokens:
            raise ValueError(
                f"attention sequence length {a.shape[2]} does not match the "
                f"{num_tokens} tokens of the prompt"
            )
        if np.any(a < 0):
            raise ValueError("attention weights must be non-negative")
        if not np.allclose(a.sum(axis=-1), 1.0, atol=1e-4):
            raise ValueError("attention rows must be normalised to sum to one")
        return a

    def step(
        self,
        code: str,
        attention: npt.ArrayLike,
        total_blocks: Optional[int] = None,
    ) -> EvictionPlan:
        """Run one observation step.

        Parameters
        ----------
        code:
            The prompt source code (or natural-language prompt).
        attention:
            Attention tensor of shape ``(L, H_Q, T)`` where ``T`` is the number
            of tokens produced for ``code``.  Typically the last-token attention
            rows (one per head) stacked across layers.
        total_blocks:
            Retention budget :math:`\\mathcal{B}_{\\text{total}}` — the number of
            physical blocks to keep resident (the eviction target).  Defaults to
            the physical capacity ``ceil(T / B)`` (no eviction).

        Returns
        -------
        EvictionPlan
            The permanent mask, budgets, and per-layer selected pages.
        """
        # Scenario 1: AST-aware semantic slicing.
        num_tokens = self.slicer.token_count(code)
        num_blocks = int(np.ceil(num_tokens / self.config.block_size)) if num_tokens else 0
        if total_blocks is None:
            total_blocks = num_blocks
        else:
            total_blocks = int(total_blocks)
            if total_blocks < 0:
                raise ValueError(f"total_blocks must be non-negative, got {total_blocks}")

        # Pinned pages are capped at gamma * B_total (the retention budget).
        pin_cap = int(np.floor(self.config.pin_ratio * total_blocks))
        page_mask = self.slicer.page_mask(code, num_blocks=num_blocks, pin_cap=pin_cap)
        pinned = page_mask.pinned_block_indices

        attention_array = self._validate_attention(attention, num_tokens)
        num_layers = attention_array.shape[0]

        # Scenario 2: smoothed Renyi-2 layer budgets.
        layer_entropies = self.allocator.update(attention_array)
        dynamic_token_budget = (total_blocks - int(pinned.size)) * self.config.block_size
        dynamic_token_budget = max(dynamic_token_budget, 0)
        budgets = self.allocator.allocate(
            dynamic_budget=dynamic_token_budget,
            num_kv_heads=self.config.num_kv_heads,
            block_size=self.config.block_size,
        )

        # Scenario 3: group pooling + paged mapping per layer.
        selected_layers: list[npt.NDArray[np.int64]] = []
        for layer in range(num_layers):
            pooled, _weights = self.pooler.pool_attention(attention_array[layer])
            importances = block_max_reduce(pooled, self.config.block_size)
            importances = pin_blocks(importances, pinned)
            k_l = int(budgets.layer_block_budgets[layer])
            selected = select_top_k_per_group(
                importances, k_per_group=k_l, pinned_block_indices=pinned
            )
            selected_layers.append(selected.selected_blocks)

        return EvictionPlan(
            page_mask=page_mask,
            layer_entropies=layer_entropies,
            layer_token_budgets=budgets.layer_token_budgets,
            layer_block_budgets=budgets.layer_block_budgets,
            selected_blocks=tuple(selected_layers),
            pinned_block_indices=pinned,
            total_blocks=total_blocks,
            num_blocks=num_blocks,
            dynamic_token_budget=int(dynamic_token_budget),
        )

    def reset(self) -> None:
        """Forget the smoothed entropy state (e.g. between requests)."""
        self.allocator.reset()


__all__ = ["EvictionPlan", "PageEntroKVConfig", "PageEntroKVEngine"]
