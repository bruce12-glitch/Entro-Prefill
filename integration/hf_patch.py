"""HuggingFace Transformers integration for Page-EntroKV.

This adapter binds the engine to ``transformers`` models:

* :func:`make_byte_span_tokenizer` aligns a fast HuggingFace tokenizer's
  sub-word offsets with the byte offsets used by tree-sitter, so the AST
  invariant mask projects onto exactly the :math:`T` model tokens.
* :func:`capture_last_token_attention` hooks the model's attention modules to
  recover the per-layer, per-query-head last-token attention distribution
  :math:`A \\in \\mathbb{R}^{L \\times H_Q \\times T}`.
* :func:`build_plan` runs the full Page-EntroKV pipeline on a prompt.
* :func:`evict_past_key_values` translates a page plan into token-level KV-cache
  pruning for the non-paged HuggingFace cache (a page is *resident* when any
  layer selects it).

``torch`` and ``transformers`` are imported lazily so this module remains
importable (and its pure helpers testable) without them installed.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from page_entrokv.ast_slicing import TokenSpan
from page_entrokv.engine import EvictionPlan, PageEntroKVConfig, PageEntroKVEngine
from page_entrokv.metrics import resident_token_mask

try:  # pragma: no cover - availability check
    import torch

    _HAS_TORCH = True
except Exception:  # noqa: BLE001 - torch is optional at import time
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "torch is required for the HuggingFace integration; install it via "
            "`pip install torch` or the project `environment.yml`."
        )


def _char_to_byte_offsets(text: str) -> list[int]:
    """Cumulative UTF-8 byte offset of every character boundary in ``text``."""
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    return offsets


def make_byte_span_tokenizer(hf_tokenizer) -> Callable[[str], list[TokenSpan]]:
    """Wrap a fast HuggingFace tokenizer as a byte-span tokenizer.

    tree-sitter reports *byte* offsets while HuggingFace offset mappings are in
    Unicode *character* offsets; this wrapper converts between the two so the
    AST invariant mask aligns exactly with the model's :math:`T` tokens.

    Special tokens (``[BOS]``, ``[CLS]``, padding, ...) carry a zero-length
    offset mapping and are mapped to the span ``(0, 0)``, which can never
    overlap a declaration span, so they are never pinned.
    """
    add_special = getattr(hf_tokenizer, "add_special_tokens", True)

    def span_tokenizer(text: str) -> list[TokenSpan]:
        encoding = hf_tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=add_special
        )
        offsets = encoding.get("offset_mapping")
        if offsets is None:
            raise ValueError(
                "The provided tokenizer does not expose `offset_mapping`; a fast "
                "tokenizer (e.g. `AutoTokenizer.from_pretrained(..., use_fast=True)`) "
                "is required for byte-accurate AST slicing."
            )
        char_to_byte = _char_to_byte_offsets(text)
        spans: list[TokenSpan] = []
        for start, end in offsets:
            start, end = int(start), int(end)
            if end <= start:
                spans.append((0, 0))
            else:
                spans.append((char_to_byte[start], char_to_byte[end]))
        return spans

    return span_tokenizer


def _attention_module_names() -> tuple[str, ...]:
    """Class-name suffixes treated as attention modules during capture.

    HuggingFace names all attention implementations with an ``Attention``
    suffix (``SelfAttention``, ``SdpaAttention``, ``FlashAttention2``, ...), so
    matching the single suffix ``Attention`` is sufficient and robust across
    versions.
    """
    return ("Attention",)


class AttentionCapture:
    """Forward-hook based capture of last-token attention weights.

    Parameters
    ----------
    engine:
        A :class:`PageEntroKVEngine` (optional; only needed if you want the
        hook to run the full pipeline on-the-fly).
    """

    def __init__(self, engine: Optional[PageEntroKVEngine] = None) -> None:
        self.engine = engine
        self.captured: list[npt.NDArray[np.float64]] = []
        self._handles: list = []

    def _hook(self, module, args, output) -> None:  # noqa: ARG002 - module unused
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            weights = output[1]
        else:
            weights = output
        if weights is None:
            raise RuntimeError(
                "Attention weights were not produced by the model forward; call "
                "the model with `output_attentions=True` (and avoid fused/SDPA "
                "implementations that drop the weight tensor)."
            )
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().float()
            # (B, H, T_q, T_k) -> (H, T_k): batch-mean, last query row.
            weights = weights.mean(dim=0)[:, -1, :]
            weights = weights.cpu().numpy()
        self.captured.append(np.asarray(weights, dtype=np.float64))

    def attach(self, model) -> None:
        """Register the capture hook on every attention sub-module."""
        _require_torch()
        self.detach()
        for module in model.modules():
            name = module.__class__.__name__
            if name.endswith(_attention_module_names()):
                self._handles.append(module.register_forward_hook(self._hook))

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "AttentionCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()

    def capture(
        self,
        model,
        input_ids,
        attention_mask=None,
        **forward_kwargs,
    ) -> npt.NDArray[np.float64]:
        """Run one forward pass and return attention of shape ``(L, H_Q, T)``."""
        _require_torch()
        self.captured = []
        self.attach(model)
        try:
            model.eval()
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    **forward_kwargs,
                )
        finally:
            self.detach()
        if not self.captured:
            raise RuntimeError("No attention modules were hooked; nothing captured.")
        return np.stack(self.captured, axis=0)


def capture_last_token_attention(
    model,
    input_ids,
    attention_mask=None,
    **forward_kwargs,
) -> npt.NDArray[np.float64]:
    """Convenience wrapper returning last-token attention ``(L, H_Q, T)``."""
    return AttentionCapture().capture(
        model, input_ids, attention_mask=attention_mask, **forward_kwargs
    )


def build_plan(
    hf_tokenizer,
    text: str,
    attention: npt.ArrayLike,
    config: Optional[PageEntroKVConfig] = None,
    total_blocks: Optional[int] = None,
) -> EvictionPlan:
    """Run the Page-EntroKV pipeline for a prompt and its captured attention.

    Parameters
    ----------
    hf_tokenizer:
        A fast HuggingFace tokenizer used for byte-accurate AST slicing.
    text:
        The raw prompt string (source code for AST slicing).
    attention:
        Attention tensor of shape ``(L, H_Q, T)`` as returned by
        :func:`capture_last_token_attention`.
    config:
        Engine configuration (defaults to :class:`PageEntroKVConfig`).
    total_blocks:
        Physical capacity; defaults to ``ceil(T / block_size)``.
    """
    cfg = config if config is not None else PageEntroKVConfig()
    span_tokenizer = make_byte_span_tokenizer(hf_tokenizer)
    engine = PageEntroKVEngine(cfg, tokenizer=span_tokenizer)
    return engine.step(text, attention, total_blocks=total_blocks)


def resident_blocks(plan: EvictionPlan) -> npt.NDArray[np.int64]:
    """Union over layers of selected blocks (pinned pages included).

    In a shared paged KV cache a physical page is resident whenever *any* layer
    selects it, so the union is the correct eviction key.
    """
    if not plan.selected_blocks:
        return plan.pinned_block_indices.copy()
    union = np.concatenate([np.asarray(b) for b in plan.selected_blocks])
    return np.sort(np.unique(union))


def evict_past_key_values(
    past_key_values,
    selected_blocks: Sequence[int],
    block_size: int,
    num_tokens: int,
):
    """Prune a HuggingFace ``past_key_values`` cache to resident blocks.

    Parameters
    ----------
    past_key_values:
        A tuple of ``(key, value)`` tensor pairs, each of shape ``(B, H, S, D)``.
    selected_blocks:
        Resident block indices (e.g. from :func:`resident_blocks`).
    block_size, num_tokens:
        Page geometry of the cache.

    Returns
    -------
    tuple
        A new tuple of ``(key, value)`` pairs sliced along the sequence axis to
        the resident tokens only.
    """
    _require_torch()
    keep = resident_token_mask(selected_blocks, block_size, num_tokens)
    keep_idx = np.flatnonzero(keep).astype(np.int64)
    pruned = []
    for layer in past_key_values:
        key, value = layer[0], layer[1]
        index = torch.as_tensor(keep_idx, dtype=torch.long, device=key.device)
        pruned.append((key[..., index, :], value[..., index, :]))
    return tuple(pruned)


__all__ = [
    "AttentionCapture",
    "build_plan",
    "capture_last_token_attention",
    "evict_past_key_values",
    "make_byte_span_tokenizer",
    "resident_blocks",
    "resident_token_mask",
]
