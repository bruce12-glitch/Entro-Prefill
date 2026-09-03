#!/usr/bin/env python3
"""RULER-style synthetic long-context benchmark for Page-EntroKV.

RULER constructs long synthetic contexts (variable tracking, common/frequent
word retrieval, multi-hop tracing, ...) and queries a small set of *key* tokens.
Page-EntroKV must keep the pages containing those key tokens resident under a
strict budget.

Each JSONL sample has a ``context`` string and a ``keys`` list of substrings
whose occurrences are the tokens that must survive.  The script reports:

* ``uor``            -- Unique Occupancy Ratio (exactly 1.00);
* ``key_retention``  -- fraction of key tokens whose page is resident;
* ``coverage``       -- fraction of physical blocks kept resident.

Backends: ``--backend synthetic`` (no ``torch`` needed) or ``--backend hf``.
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
    resident_token_mask,
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
    key_tokens: np.ndarray,
) -> np.ndarray:
    """Deterministic attention peaking on the key tokens for focused heads."""
    attention = np.zeros((num_layers, num_query_heads, num_tokens), dtype=np.float64)
    for layer in range(num_layers):
        for head in range(num_query_heads):
            if rng.random() < 0.5:
                focus = np.union1d(key_tokens, rng.choice(num_tokens, size=3, replace=False))
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


def locate_key_tokens(engine: PageEntroKVEngine, text: str, keys: list[str]) -> np.ndarray:
    """Token indices overlapping any occurrence of any key substring.

    Key occurrences are collected once per key, then swept against the token
    spans in a single pass.
    """
    spans = engine.slicer.tokenize(text)
    if not keys:
        return np.array([], dtype=np.int64)

    occurrences: list[tuple[int, int]] = []
    for key in keys:
        if not key:
            continue
        start = 0
        while True:
            pos = text.find(key, start)
            if pos < 0:
                break
            occurrences.append((pos, pos + len(key)))
            start = pos + 1
    if not occurrences:
        return np.array([], dtype=np.int64)

    kept = [
        i
        for i, (s, e) in enumerate(spans)
        if any(s < occ_end and e > occ_start for occ_start, occ_end in occurrences)
    ]
    return np.array(kept, dtype=np.int64)


def run_one(
    engine: PageEntroKVEngine,
    text: str,
    attention: np.ndarray,
    keys: list[str],
    total_blocks: int,
) -> dict:
    plan = engine.step(text, attention, total_blocks=total_blocks)
    selected = [np.asarray(arr, dtype=np.int64) for arr in plan.selected_blocks]
    key_tokens = locate_key_tokens(engine, text, keys)

    if selected:
        union = np.unique(np.concatenate(selected))
        keep = resident_token_mask(union, plan.page_mask.block_size, plan.page_mask.num_tokens)
    else:
        union = np.array([], dtype=np.int64)
        keep = np.zeros(plan.page_mask.num_tokens, dtype=bool)

    per_layer_uor = (
        [unique_occupancy_ratio(arr, plan.num_blocks) for arr in selected]
        if selected
        else [0.0]
    )
    key_retention = float(keep[key_tokens].mean()) if key_tokens.size else 1.0
    return {
        "num_tokens": plan.page_mask.num_tokens,
        "num_blocks": plan.num_blocks,
        "budget_blocks": plan.total_blocks,
        "pinned_blocks": plan.pinned_block_indices.tolist(),
        "key_tokens": int(key_tokens.size),
        "uor": float(np.mean(per_layer_uor)),
        "coverage": float(union.size) / max(1, plan.num_blocks),
        "key_retention": key_retention,
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
    parser.add_argument("--output", type=Path, default=Path("ruler_report.json"))
    args = parser.parse_args(argv)

    config = _engine_config(args)
    engine = PageEntroKVEngine(config)
    rng = np.random.default_rng(args.seed)
    samples = load_samples(args.data)

    results = []
    for index, sample in enumerate(samples):
        text = sample["context"]
        keys = sample.get("keys", [])
        num_tokens = engine.slicer.token_count(text)
        physical = int(np.ceil(num_tokens / args.block_size)) if num_tokens else 0
        total_blocks = max(1, int(np.floor(args.retention * physical))) if physical else 0
        if args.backend == "hf":
            attention = _real_attention(args, text, num_tokens)
        else:
            key_tokens = locate_key_tokens(engine, text, keys)
            attention = synthetic_attention(
                args.num_layers, config.num_query_heads, num_tokens, rng, key_tokens
            )
        row = run_one(engine, text, attention, keys, total_blocks)
        row["sample"] = index
        results.append(row)
        engine.reset()

    summary = {
        "backend": args.backend,
        "num_samples": len(results),
        "mean_uor": float(np.mean([r["uor"] for r in results])),
        "mean_coverage": float(np.mean([r["coverage"] for r in results])),
        "mean_key_retention": float(np.mean([r["key_retention"] for r in results])),
    }
    args.output.write_text(
        json.dumps({"summary": summary, "samples": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
