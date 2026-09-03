# Page-EntroKV

**Entropy-guided, AST-aware KV-page selection for Grouped-Query Attention (GQA)
paged serving engines.**

Page-EntroKV resolves the structural mismatch between *dynamic KV-cache
eviction* (which reasons about logical query heads and token positions) and the
*physical* GQA memory layout of PagedAttention (which stores a shared set of
fixed-size physical blocks). It pins AST invariant declarations as *permanent*
anchor pages, allocates the remaining *dynamic* page budget across layers with a
discrete Rényi-2 (collision) entropy, pools logical query-head attention onto
physical KV heads, and selects pages with a unique-occupancy guarantee.

---

## Overview

Long-context serving is bounded by KV-cache memory. Two structural facts
dominate the design space:

1. **GQA shares KV heads.** With query heads `H_Q` and KV heads `H_KV`, the
   group ratio `r = H_Q / H_KV` means `r` logical query heads share one physical
   KV cache, so a per-*logical-head* eviction score is not directly realizable
   on physical memory.
2. **PagedAttention stores fixed-size blocks.** The KV cache lives in physical
   blocks `P_b` of size `B` shared across layers; eviction must therefore be a
   *page-level* decision with no intra-page fragmentation.

Page-EntroKV bridges both with three cooperating mechanisms.

| Scenario | Module | Role |
| --- | --- | --- |
| 1. AST-aware semantic slicing | `page_entrokv/ast_slicing.py` | Permanent anchor-page mask for invariant declarations |
| 2. Rényi-2 layer budgets | `page_entrokv/entropy_budget.py` | Distribute the dynamic budget across layers |
| 3. GQA pooling + paged mapping | `page_entrokv/group_pooler.py`, `page_entrokv/paged_mapping.py` | Map pooled attention to physical pages with UOR ≡ 1.00 |

---

## Method

### Scenario 1 — AST-aware semantic slicing

Given a source-code prompt, tree-sitter identifies the invariant declaration
vertices

```
V_def = { class_definition, function_definition, type_annotation,
          import_statement, top-level global scopes }
```

whose byte spans are projected onto token offsets to obtain the invariant token
set

```
T_AST = ⋃_{v ∈ V_def} span(v).
```

The permanent physical page mask over blocks `P_b` of size `B` is

```
M_AST(P_b) = 1  iff  tokens(P_b) ∩ T_AST ≠ ∅   (0 otherwise).
```

A dynamic cap bounds the anchor budget,

```
B_pinned ≤ γ · B_total     (default γ = 0.30),
```

so anchor pages can never exhaust eviction capacity. For natural-language
prompts (or when no parser is available) the slicer degrades gracefully to an
empty mask.

### Scenario 2 — Discrete Rényi-2 layer budget allocation

Let `A_{l,h} ∈ R^T` be the attention distribution of logical head `h` at layer
`l`. The discrete collision (Rényi-2, `α = 2`) entropy is

```
H_2(A_{l,h}) = -ln Σ_t (A_{l,h}(t))² = -ln ‖A_{l,h}‖₂².
```

The normalizer `ε = 10⁻¹²` is applied *inside* the logarithm only
(`-ln max(‖A‖₂², ε)`), preserving `Σ_t A(t) = 1` exactly and keeping the
entropy non-negative. The layer-mean entropy is

```
H̄_2^(l) = (1/H_Q) Σ_h H_2(A_{l,h}),
```

and the dynamic budget (total budget minus the pinned anchor budget) is
allocated per layer by a Boltzmann rule,

```
B_dynamic = B_total − |P_pinned| · B,
B_l = max(1, ⌊ B_dynamic · exp(−H̄_2^(l)/τ_L) / Σ_j exp(−H̄_2^(j)/τ_L) ⌋).
```

Layers are allocated over an observation window `W_obs` (EMA smoothing) so a
single decode step cannot thrash the budgets.

### Scenario 3 — GQA group pooling & paged-frame mapping

For group ratio `r = H_Q / H_KV`, physical KV head `g` owns query heads
`Group(g) = {h : ⌊h/r⌋ = g}`. Intra-group Boltzmann weights based on individual
head entropies,

```
w_{l,h} = exp(−H_2(A_{l,h})/τ_g) / Σ_{j ∈ Group(g)} exp(−H_2(A_{l,j})/τ_g),
```

pool the group distribution

```
A_{l,g}^pooled(t) = Σ_{h ∈ Group(g)} w_{l,h} · A_{l,h}(t).
```

The Block-Max reduction projects onto physical pages,

```
I_{l,g}(P_b) = max_{t ∈ tokens(P_b)} A_{l,g}^pooled(t),
```

anchor pages are pinned with `I_{l,g}(P_b) = +∞`, and each group selects its top

```
K_l = ⌊ B_l / (H_KV · B) ⌋
```

blocks. Selection **strictly enforces UOR ≡ 1.00** — no physical block is ever
selected by two KV groups, so the multi-layer page table contains zero
duplicate pages.

---

## Installation

```bash
# Minimal (pure-NumPy core + tree-sitter):
pip install -e .

# Full stack (torch + transformers for real-model runs):
conda env create -f environment.yml
conda activate entrokv
```

The heavy dependencies (`torch`, `transformers`, `vllm`) are optional: the
library and benchmarks run in a dependency-light *synthetic* backend without
them.

## Quickstart

```python
import numpy as np
from page_entrokv import PageEntroKVConfig, PageEntroKVEngine

code = (
    "import os\n"
    "from typing import List\n"
    "LIMIT = 128\n\n"
    "def lookup(key: str) -> List[int]:\n"
    "    return [ord(c) for c in key]\n"
)

config = PageEntroKVConfig(
    block_size=16,          # B: tokens per page
    pin_ratio=0.30,         # gamma: pinned-page cap
    num_kv_heads=8,         # H_KV
    num_query_heads=32,     # H_Q
)
engine = PageEntroKVEngine(config)

T = engine.slicer.token_count(code)
L = 12
rng = np.random.default_rng(0)
attention = rng.random((L, config.num_query_heads, T))
attention /= attention.sum(axis=-1, keepdims=True)

plan = engine.step(code, attention)
print("pinned anchor pages:", plan.pinned_block_indices)
print("layer block budgets  :", plan.layer_block_budgets)
print("selected (layer 0)   :", plan.selected_blocks[0])
```

`plan.selected_blocks` is the per-layer list of resident physical pages (each
sorted and duplicate-free). Feed those block indices into your serving engine's
block manager to evict the complement.

## Tests

```bash
python -m pytest tests -v
```

The suite includes property tests fuzzing the selection across hundreds of
random configurations and asserting the **Unique Occupancy Ratio identity
(UOR ≡ 1.00)** holds exactly.

## Benchmarks

```bash
bash scripts/reproduce_paper.sh all
```

Individual stages: `test`, `niah`, `longbench`, `ruler`, `install`. Set
`ENTROKV_DATA_DIR` to a directory containing `niah.jsonl`, `longbench.jsonl`,
and `ruler.jsonl` corpora; use `ENTROKV_BACKEND=hf ENTROKV_MODEL=...` to run
with real model attention, or leave `ENTROKV_BACKEND=synthetic` for the
dependency-light smoke run.

## Integration

* **HuggingFace** — `integration/hf_patch.py` captures last-token attention via
  forward hooks, aligns sub-word offsets to tree-sitter byte spans, and prunes
  `past_key_values` to resident pages.
* **vLLM** — `integration/vllm_worker_patch.py` registers attention-capture
  hooks on the PagedAttention worker and translates an `EvictionPlan` into a
  physical block-table mask.

## Repository layout

```
page-entrokv/
├── page_entrokv/        # core library (ast_slicing, entropy_budget,
│                        #   group_pooler, paged_mapping, engine, metrics)
├── integration/         # HF + vLLM adapters
├── scripts/             # NIAH / LongBench / RULER benchmarks + reproduction
├── tests/               # unit + property tests (incl. UOR identity)
├── environment.yml      # full conda environment
├── requirements.txt     # minimal pip requirements
├── setup.py             # packaging
└── .github/workflows/   # CI
```

## Citation

```bibtex
@misc{pageentrokv,
  title        = {Page-EntroKV: Entropy-Guided, AST-Aware KV-Page Selection
                  for GQA Paged Attention},
  author       = {Page-EntroKV contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/bruce12-glitch/Entro-Prefill}},
}
```

## License

Apache-2.0.
