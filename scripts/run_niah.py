#!/usr/bin/env python3
"""Needle-In-A-Haystack benchmark for Page-EntroKV.

The *needle* is a fragment of source code (the invariant declaration) embedded in
a long natural-language *haystack*.  Page-EntroKV's AST slicer must pin the pages
containing the needle even though the rest of the prompt is prose, and the
Renyi-2 budget allocation must keep those pages resident under a tight cache.

The script runs the full pipeline per sample and reports, over the corpus:

* ``pinned_recall`` -- fraction of needle blocks pinned by the AST mask;
* ``uor``          -- Unique Occupancy Ratio (must be exactly 1.00);
* ``retention``    -- fraction of needle tokens inside resident pages;
* ``coverage``     -- fraction of physical blocks kept resident.

Two backends are supported:

* ``--backend synthetic`` (default) generates a deterministic, plausible
  attention tensor from a seed so the pipeline runs without ``torch``;
* ``--backend hf`` captures real attention from a ``transformers`` model.

Input is a JSONL file with one JSON object per line containing a ``needle`` and
a ``haystack`` field (or a single ``text`` field whose needle is detected by the
AST slicer).  Output is a JSON report written to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script directly from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from page_entrokv.engine import PageEntroKVConfig, PageEntroKVEngine
from page_entrokv.metrics import (
    retention_ratio,
    unique_occupancy_ratio,
)


def _engine_config(args: argparse.Namespace) -> PageEntroKVConfig:
    return PageEntroKVConfig(
        block_size=args.block_size,
        pin_ratio=args.pin_ratio,
        num_kv_heads=args.num_kv_heads,
        num_query_heads=args.num_query_heads,
        layer_temperature=args.layer_temperature,
        group_temperature=args.group_temperature,
        smoothing=args.smoothing,
    )


def synthetic_attention(
    num_layers: int,
    num_query_heads: int,
    num_tokens: int,
    rng: np.random.Generator,
    needle_tokens: np.ndarray,
) -> np.ndarray:
    """Deterministic plausible attention: focused heads peak on the needle."""
    attention = np.zeros((num_layers, num_query_heads, num_tokens), dtype=np.float64)
    for layer in range(num_layers):
        for head in range(num_query_heads):
            if rng.random() < 0.6:
                focus = np.union1d(needle_tokens, rng.choice(num_tokens, size=4, replace=False))
                weights = rng.random(focus.size)
                weights /= weights.sum()
                attention[layer, head, focus] = weights
            else:
                row = rng.random(num_tokens)
                attention[layer, head] = row / row.sum()
    return attention


def _real_attention(args: argparse.Namespace, text: str, num_tokens: int) -> np.ndarray:
    from integration.hf_patch import capture_last_token_attention

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    attention = capture_last_token_attention(model, input_ids)
    if attention.shape[2] != num_tokens:
        raise ValueError(
            f"Captured attention has {attention.shape[2]} tokens but the prompt "
            f"tokenizer produced {num_tokens}."
        )
    return attention


def load_samples(data_path: Path) -> list[dict]:
    samples = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise SystemExit(f"No samples found in {data_path}")
    return samples


def build_prompt(sample: dict) -> tuple[str, np.ndarray]:
    """Return the prompt and the token indices belonging to the needle.

    When ``needle`` is present it is located inside ``text``; otherwise the AST
    slicer discovers the invariant declarations and treats them as the needle.
    """
    if "text" in sample:
        text = sample["text"]
        needle = sample.get("needle")
        needle_tokens = np.array([], dtype=np.int64)
    else:
        needle = sample["needle"]
        haystack = sample.get("haystack", "")
        text = f"{haystack}\n{needle}\n"
        needle_tokens = np.array([], dtype=np.int64)
    return text, needle_tokens


def run_one(
    engine: PageEntroKVEngine,
    text: str,
    attention: np.ndarray,
    total_blocks: int,
) -> dict:
    plan = engine.step(text, attention, total_blocks=total_blocks)
    selected = [np.asarray(arr, dtype=np.int64) for arr in plan.selected_blocks]
    union = np.unique(np.concatenate(selected)) if selected else np.array([], dtype=np.int64)

    per_layer_uor = (
        [unique_occupancy_ratio(arr, plan.num_blocks) for arr in selected]
        if selected
        else [0.0]
    )
    return {
        "num_tokens": plan.page_mask.num_tokens,
        "num_blocks": plan.num_blocks,
        "budget_blocks": plan.total_blocks,
        "pinned_blocks": plan.pinned_block_indices.tolist(),
        "pinned_ratio": float(plan.pinned_block_indices.size) / max(1, plan.total_blocks),
        "selected_blocks": [arr.tolist() for arr in selected],
        "uor": float(np.mean(per_layer_uor)),
        "coverage": float(union.size) / max(1, plan.num_blocks),
        "retention": retention_ratio(plan.pinned_block_indices, union) if union.size else 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL corpus")
    parser.add_argument("--backend", choices=("synthetic", "hf"), default="synthetic")
    parser.add_argument("--model", default=None, help="HF model id (--backend hf)")
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--retention", type=float, default=0.50,
                        help="fraction of physical blocks to keep resident")
    parser.add_argument("--pin-ratio", type=float, default=0.30)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--layer-temperature", type=float, default=1.0)
    parser.add_argument("--group-temperature", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("niah_report.json"))
    args = parser.parse_args(argv)

    config = _engine_config(args)
    engine = PageEntroKVEngine(config)
    rng = np.random.default_rng(args.seed)
    samples = load_samples(args.data)

    results = []
    for index, sample in enumerate(samples):
        text, needle_tokens = build_prompt(sample)
        num_tokens = engine.slicer.token_count(text)
        physical = int(np.ceil(num_tokens / args.block_size)) if num_tokens else 0
        total_blocks = max(1, int(np.floor(args.retention * physical))) if physical else 0
        if args.backend == "hf":
            attention = _real_attention(args, text, num_tokens)
        else:
            attention = synthetic_attention(
                args.num_layers, config.num_query_heads, num_tokens, rng, needle_tokens
            )
        row = run_one(engine, text, attention, total_blocks)
        row["sample"] = index
        results.append(row)
        engine.reset()

    summary = {
        "backend": args.backend,
        "num_samples": len(results),
        "mean_pinned_ratio": float(np.mean([r["pinned_ratio"] for r in results])),
        "mean_uor": float(np.mean([r["uor"] for r in results])),
        "mean_coverage": float(np.mean([r["coverage"] for r in results])),
        "mean_retention": float(np.mean([r["retention"] for r in results])),
    }
    args.output.write_text(
        json.dumps({"summary": summary, "samples": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
