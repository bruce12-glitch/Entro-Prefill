"""Integration adapters binding Page-EntroKV to real serving stacks.

* :mod:`integration.hf_patch` -- HuggingFace Transformers: attention capture,
  byte-span tokenizer alignment, and token-level KV pruning.
* :mod:`integration.vllm_worker_patch` -- vLLM: attention capture hooks and
  block-table masking for the PagedAttention worker.

Both modules import their heavy dependencies (``torch``, ``transformers``,
``vllm``) lazily, so they are always importable and safe to unit-test in
isolation.
"""

from __future__ import annotations

from integration import hf_patch, vllm_worker_patch

__all__ = ["hf_patch", "vllm_worker_patch"]
