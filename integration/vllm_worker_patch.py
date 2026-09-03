"""vLLM PagedAttention worker integration for Page-EntroKV.

vLLM stores KV caches in *physical* blocks shared across layers and KV heads.
This module supplies the glue that runs the Page-EntroKV selector on a vLLM
worker and translates the resulting plan into a physical block-table mask:

* :class:`PageEntroKVSelector` wraps a :class:`PageEntroKVEngine` and computes
  :class:`EvictionPlan` objects from a prompt and its captured attention.
* :func:`install_attention_capture` registers a forward hook that captures the
  per-layer last-token attention during ``execute_model``.
* :func:`block_table_mask` converts an :class:`EvictionPlan` into a boolean mask
  over physical block indices, marking which blocks to keep resident.
* :func:`apply_mask_to_block_table` rewrites a vLLM block table in place.

``torch`` and ``vllm`` are imported lazily so the module is importable (and its
pure helpers testable) without them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from page_entrokv.engine import EvictionPlan, PageEntroKVConfig, PageEntroKVEngine

try:  # pragma: no cover - availability check
    import torch

    _HAS_TORCH = True
except Exception:  # noqa: BLE001 - torch is optional at import time
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "torch is required for the vLLM integration; install it via "
            "`pip install torch` or the project `environment.yml`."
        )


class PageEntroKVSelector:
    """Engine wrapper producing eviction plans for a vLLM worker.

    Parameters
    ----------
    config:
        Engine configuration.
    tokenizer:
        Byte-span tokenizer callable (see
        :func:`integration.hf_patch.make_byte_span_tokenizer`).  When ``None``
        the engine's word-level tokenizer is used, which requires the prompt
        text to be token-aligned with the model.
    parser:
        Optional pre-built tree-sitter parser.
    """

    def __init__(
        self,
        config: Optional[PageEntroKVConfig] = None,
        tokenizer=None,
        parser=None,
    ) -> None:
        self.engine = PageEntroKVEngine(config, tokenizer=tokenizer, parser=parser)
        self._captured: list[npt.NDArray[np.float64]] = []
        self._hooks: list = []

    @property
    def last_plan(self) -> Optional[EvictionPlan]:
        """The most recent plan, if any."""
        return self._last_plan

    def select(
        self,
        prompt: str,
        attention: npt.ArrayLike,
        total_blocks: Optional[int] = None,
    ) -> EvictionPlan:
        """Run one engine step and store the result as ``last_plan``."""
        plan = self.engine.step(prompt, attention, total_blocks=total_blocks)
        self._last_plan = plan
        return plan

    def _attention_hook(self, module, args, output) -> None:  # noqa: ARG002
        """Hook capturing attention from a vLLM attention module forward pass.

        vLLM attention modules return the output tensor only in most versions,
        so the hook expects the attention weights to be attached to the output
        as ``output.attn_weights`` by the caller's patched attention kernel, or
        passed through ``kwargs``.  Both conventions are supported.
        """
        weights = None
        if isinstance(output, tuple) and len(output) >= 2:
            weights = output[1]
        elif isinstance(output, torch.Tensor):
            weights = getattr(output, "attn_weights", None)
        if weights is None:
            return  # no weights available on this step; skip rather than fail.
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().float()
            if weights.ndim == 4:  # (B, H, T_q, T_k)
                weights = weights.mean(dim=0)[:, -1, :]
            elif weights.ndim == 3:  # (H, T_q, T_k)
                weights = weights[:, -1, :]
            weights = weights.cpu().numpy()
        self._captured.append(np.asarray(weights, dtype=np.float64))

    def install_attention_capture(self, model) -> None:
        """Register attention capture hooks on a vLLM model runner."""
        _require_torch()
        self.remove_attention_capture()
        for module in model.modules():
            if module.__class__.__name__ in (
                "Attention",
                "PagedAttention",
                "AttentionImpl",
                "GPTQAttention",
            ):
                self._hooks.append(module.register_forward_hook(self._attention_hook))

    def remove_attention_capture(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def captured_attention(self) -> npt.NDArray[np.float64]:
        """Return the stacked captured attention ``(L, H_Q, T)``."""
        if not self._captured:
            raise RuntimeError("No attention captured yet.")
        return np.stack(self._captured, axis=0)

    def __enter__(self) -> "PageEntroKVSelector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove_attention_capture()


def install_attention_capture(model, selector: PageEntroKVSelector) -> PageEntroKVSelector:
    """Register attention capture hooks on ``model`` and return ``selector``."""
    selector.install_attention_capture(model)
    return selector


def block_table_mask(
    plan: EvictionPlan,
    num_blocks: Optional[int] = None,
) -> npt.NDArray[np.bool_]:
    """Boolean mask over physical blocks marking which pages stay resident.

    A page is resident when any layer selects it (the union of the per-layer
    selections, which already contains the pinned anchor pages).
    """
    if num_blocks is None:
        num_blocks = plan.total_blocks
    mask = np.zeros(num_blocks, dtype=bool)
    if plan.selected_blocks:
        union = np.concatenate([np.asarray(b) for b in plan.selected_blocks])
        union = np.unique(union)
        if union.size and union.max() >= num_blocks:
            raise IndexError(
                f"plan references block {union.max()} but only {num_blocks} exist"
            )
        mask[union] = True
    return mask


def apply_mask_to_block_table(
    block_table,
    mask: npt.ArrayLike,
) -> None:
    """Zero out (evict) non-resident entries of a vLLM block table in place.

    ``block_table`` is a sequence (per sequence in the batch) of integer block
    indices.  Entries whose block is not in ``mask`` are set to ``-1`` so the
    scheduler will not reuse (or will recompute) those slots.
    """
    keep = np.asarray(mask, dtype=bool)
    for sequence in block_table:
        for slot in range(len(sequence)):
            block = int(sequence[slot])
            if block >= 0 and (block >= keep.size or not keep[block]):
                sequence[slot] = -1


def resident_blocks_of(plan: EvictionPlan) -> npt.NDArray[np.int64]:
    """Union over layers of selected blocks (convenience alias)."""
    from integration.hf_patch import resident_blocks

    return resident_blocks(plan)


__all__ = [
    "PageEntroKVSelector",
    "apply_mask_to_block_table",
    "block_table_mask",
    "install_attention_capture",
    "resident_blocks_of",
]
