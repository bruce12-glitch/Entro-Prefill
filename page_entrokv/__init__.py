"""Page-EntroKV: entropy-guided, AST-aware KV-page selection for GQA paged attention.

Page-EntroKV resolves the structural mismatch between dynamic KV-cache eviction and
physical Grouped-Query Attention (GQA) memory layouts in PagedAttention serving
engines.  It combines

1. AST-aware semantic slicing (``ast_slicing``) that pins *permanent* pages
   containing invariant declaration tokens,
2. discrete Renyi-2 (collision) entropy budget allocation (``entropy_budget``)
   that spreads the dynamic page budget across transformer layers,
3. GQA group pooling (``group_pooler``) and paged-frame block-max mapping
   (``paged_mapping``) that translate logical query-head attention into
   per-physical-page importance scores and a unique-occupancy selection.

The public entry point is :class:`page_entrokv.engine.PageEntroKVEngine`.
"""

from __future__ import annotations

from page_entrokv.ast_slicing import (
    ASTSlicer,
    DECLARATION_NODE_TYPES,
    GLOBAL_SCOPE_NODE_TYPES,
    PageMask,
    TokenSpan,
    collect_declaration_spans,
    default_tokenize,
    invariant_token_mask,
    page_mask_from_tokens,
)
from page_entrokv.engine import (
    EvictionPlan,
    PageEntroKVConfig,
    PageEntroKVEngine,
)
from page_entrokv.entropy_budget import (
    EntropyBudgetAllocator,
    LayerBudget,
    allocate_layer_token_budgets,
    blocks_per_group,
    collision_entropies,
    collision_entropy,
    layer_mean_entropies,
    softmax_layer_weights,
)
from page_entrokv.group_pooler import (
    GroupPooler,
    group_index,
    intra_group_weights,
    pool_attention,
)
from page_entrokv.paged_mapping import (
    PagedSelection,
    block_max_reduce,
    block_sum_reduce,
    pin_blocks,
    select_blocks,
    select_top_k_per_group,
)
from page_entrokv import metrics

__version__ = "0.1.0"

__all__ = [
    "ASTSlicer",
    "DECLARATION_NODE_TYPES",
    "GLOBAL_SCOPE_NODE_TYPES",
    "PageMask",
    "TokenSpan",
    "collect_declaration_spans",
    "default_tokenize",
    "invariant_token_mask",
    "page_mask_from_tokens",
    "EvictionPlan",
    "PageEntroKVConfig",
    "PageEntroKVEngine",
    "EntropyBudgetAllocator",
    "LayerBudget",
    "allocate_layer_token_budgets",
    "blocks_per_group",
    "collision_entropies",
    "collision_entropy",
    "layer_mean_entropies",
    "softmax_layer_weights",
    "GroupPooler",
    "group_index",
    "intra_group_weights",
    "pool_attention",
    "PagedSelection",
    "block_max_reduce",
    "block_sum_reduce",
    "pin_blocks",
    "select_blocks",
    "select_top_k_per_group",
    "metrics",
    "__version__",
]
