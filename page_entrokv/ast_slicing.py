"""AST-aware semantic slicing for Page-EntroKV (Scenario 1).

Theory
------
Given a source-code prompt, we parse it with ``tree-sitter`` and identify the set
of *invariant declaration vertices*

.. math::
    \\mathcal{V}_{\\text{def}} = \\{
        \\text{class\\_definition}, \\text{function\\_definition},
        \\text{type\\_annotation}, \\text{import\\_statement},
        \\text{top-level global scopes}
    \\}.

Each vertex :math:`v` carries a byte span :math:`\\text{span}(v) = [s_v, e_v)` in
the source text.  Projecting those spans onto a tokenization of the prompt yields
the invariant token set

.. math::
    \\mathcal{T}_{\\text{AST}} = \\bigcup_{v \\in \\mathcal{V}_{\\text{def}}}
        \\text{span}(v).

For PagedAttention physical blocks :math:`P_b` of size :math:`B` we then build the
*permanent* page mask

.. math::
    \\mathcal{M}_{\\text{AST}}(P_b) =
    \\begin{cases}
        1 & \\text{if } \\text{tokens}(P_b) \\cap \\mathcal{T}_{\\text{AST}}
              \\neq \\emptyset \\\\
        0 & \\text{otherwise}
    \\end{cases}.

A dynamic cap :math:`\\mathcal{B}_{\\text{pinned}} \\le \\gamma
\\mathcal{B}_{\\text{total}}` (default :math:`\\gamma = 0.30`) bounds the number of
pinned anchor pages so they never exhaust dynamic eviction capacity.  When the
parser is unavailable, the parse yields no declarations (e.g. natural-language
prompts), or the prompt is empty, the slicer degrades gracefully to an all-zero
(empty) mask.

Implementation note
-------------------
``tree-sitter-python`` names its type-annotation node ``type``; we therefore keep
*both* ``type`` and ``type_annotation`` in :data:`DECLARATION_NODE_TYPES` for
cross-grammar compatibility.  Top-level global scopes are realised as the direct
children of the root ``module``/``program`` node.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: A half-open byte-offset span ``(start_byte, end_byte)`` in a source string.
TokenSpan = tuple[int, int]

#: Syntax-node types treated as invariant declarations.  ``type`` is the
#: tree-sitter-python name for a type annotation; ``type_annotation`` is kept for
#: grammars that use the more descriptive name.
DECLARATION_NODE_TYPES: frozenset[str] = frozenset(
    {
        "class_definition",
        "function_definition",
        "decorated_definition",
        "type",
        "type_annotation",
        "import_statement",
        "import_from_statement",
    }
)

#: Direct children of the root node that denote top-level global scopes
#: (module-level bindings such as constants, configuration, and declarations).
GLOBAL_SCOPE_NODE_TYPES: frozenset[str] = frozenset(
    {
        "expression_statement",
        "assignment",
        "class_definition",
        "function_definition",
        "decorated_definition",
        "import_statement",
        "import_from_statement",
    }
)

_DEFAULT_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def default_tokenize(code: str) -> list[TokenSpan]:
    """Tokenize ``code`` into word / punctuation byte spans.

    This is a dependency-free stand-in for a HuggingFace ``Tokenizer``.  The spans
    are half-open ``[start, end)`` byte offsets so they can be intersected with
    tree-sitter byte spans directly.  Production deployments should pass a
    tokenizer whose offsets align with the model's vocabulary via the ``tokenizer``
    argument of :class:`ASTSlicer`.
    """
    return [(m.start(), m.end()) for m in _DEFAULT_TOKEN_RE.finditer(code)]


def _spans_overlap(a: TokenSpan, b: TokenSpan) -> bool:
    """Return ``True`` iff half-open spans ``a`` and ``b`` intersect."""
    return a[0] < b[1] and b[0] < a[1]


def invariant_token_mask(
    token_spans: Sequence[TokenSpan],
    declaration_spans: Sequence[TokenSpan],
) -> npt.NDArray[np.bool_]:
    """Project declaration spans onto token indices.

    Returns a boolean mask of shape ``(T,)`` where ``T == len(token_spans)`` and
    entry ``t`` is ``True`` iff ``token_spans[t]`` intersects any declaration
    span.  Declaration spans are merged first, giving an
    :math:`O(T + |\\mathcal{V}_{\\text{def}}| \\log |\\mathcal{V}_{\\text{def}}|)`
    sweep instead of the naive quadratic scan.
    """
    mask = np.zeros(len(token_spans), dtype=np.bool_)
    if not declaration_spans or not token_spans:
        return mask

    # Merge overlapping declaration spans into disjoint intervals.
    merged: list[TokenSpan] = []
    for start, end in sorted(declaration_spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    cursor = 0
    for i, (tok_start, tok_end) in enumerate(token_spans):
        while cursor < len(merged) and merged[cursor][1] <= tok_start:
            cursor += 1
        if cursor < len(merged) and merged[cursor][0] < tok_end:
            mask[i] = True
    return mask


def page_mask_from_tokens(
    token_mask: npt.NDArray[np.bool_],
    block_size: int,
    num_blocks: Optional[int] = None,
    pin_ratio: float = 0.30,
    pin_cap: Optional[int] = None,
) -> tuple[npt.NDArray[np.bool_], bool]:
    """Reduce a token-level invariant mask to a capped page-level mask.

    Parameters
    ----------
    token_mask:
        Boolean mask of shape ``(T,)``.
    block_size:
        Number of tokens per physical page :math:`B`.
    num_blocks:
        Number of physical pages :math:`N`.  Defaults to ``ceil(T / block_size)``.
    pin_ratio:
        Cap ratio :math:`\\gamma \\in [0, 1]`.  When ``pin_cap`` is ``None`` the
        maximum number of pinned pages is ``floor(gamma * num_blocks)``.
    pin_cap:
        Explicit absolute cap :math:`\\mathcal{B}_{\\text{pinned}}` (e.g.
        ``floor(gamma * B_total)`` when the retention budget differs from the
        physical capacity).  Overrides the ``pin_ratio * num_blocks`` default.

    Returns
    -------
    page_mask:
        Boolean mask of shape ``(num_blocks,)``.
    capped:
        ``True`` iff the cap was actually applied.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not 0.0 <= pin_ratio <= 1.0:
        raise ValueError(f"pin_ratio must be in [0, 1], got {pin_ratio}")

    token_count = int(token_mask.shape[0])
    if num_blocks is None:
        num_blocks = int(np.ceil(token_count / block_size)) if token_count else 0
    num_blocks = int(num_blocks)
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")

    density = np.zeros(num_blocks, dtype=np.int64)
    for block_index in range(num_blocks):
        start = block_index * block_size
        end = min(start + block_size, token_count)
        if start < token_count:
            density[block_index] = int(token_mask[start:end].sum())

    page_mask = density > 0
    if pin_cap is None:
        max_pinned = int(np.floor(pin_ratio * num_blocks))
    else:
        max_pinned = int(pin_cap)
        if max_pinned < 0:
            raise ValueError(f"pin_cap must be non-negative, got {max_pinned}")
    max_pinned = min(max_pinned, num_blocks)
    capped = bool(int(page_mask.sum()) > max_pinned)

    if capped:
        # Keep the `max_pinned` densest pages; ``stable`` argsort breaks density
        # ties in favour of the earliest (lowest-index) page.
        order = np.argsort(-density, kind="stable")
        keep = order[:max_pinned]
        page_mask = np.zeros(num_blocks, dtype=np.bool_)
        page_mask[keep] = True

    return page_mask, capped


@dataclass
class PageMask:
    """Result of AST-aware semantic slicing for one prompt."""

    token_mask: npt.NDArray[np.bool_]
    """Token-level invariant mask, shape ``(T,)``."""

    page_mask: npt.NDArray[np.bool_]
    """Capped page-level mask, shape ``(num_blocks,)``."""

    pinned_block_indices: npt.NDArray[np.int64]
    """Indices of pinned pages, shape ``(P,)`` with ``P <= floor(gamma * N)``."""

    block_size: int
    """Tokens per page :math:`B`."""

    num_tokens: int
    """Number of tokens :math:`T`."""

    num_blocks: int
    """Number of physical pages :math:`\\mathcal{B}_{\\text{total}}`."""

    pin_ratio: float
    """The cap ratio :math:`\\gamma` that was applied."""

    pin_cap: int
    """The absolute pinned-page cap that was applied."""

    capped: bool
    """``True`` iff the cap reduced the set of pinned pages."""

    degraded: bool
    """``True`` iff the slicer fell back to an empty mask (no parser, no
    declarations, or empty input)."""


def _load_from_language_pack(language: str):
    """Build a parser with ``tree-sitter-language-pack`` (newer tree-sitter)."""
    from tree_sitter_language_pack import get_parser  # type: ignore[import-not-found]

    return get_parser(language)


def _load_from_tree_sitter_languages(language: str):
    """Build a parser with ``tree-sitter-languages`` (offline, tree-sitter <0.22)."""
    import tree_sitter_languages  # type: ignore[import-not-found]

    # ``tree-sitter-languages`` emits a deprecation warning on construction of
    # its bundled languages; it is internal to the third-party package and is
    # not actionable here, so silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return tree_sitter_languages.get_parser(language)


def _try_build_parser(language: str):
    """Return a tree-sitter parser for ``language`` or ``None`` if unavailable.

    The function never raises: a missing parser only disables the AST heuristic,
    which is the documented graceful-degradation path.
    """
    loaders = (_load_from_language_pack, _load_from_tree_sitter_languages)
    for loader in loaders:
        try:
            parser = loader(language)
            if parser is not None:
                return parser
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            logger.debug("tree-sitter loader %s failed: %s", loader.__name__, exc)
    logger.warning(
        "tree-sitter parser for %r is unavailable; AST slicing degrades to an "
        "empty (natural-language) mask.",
        language,
    )
    return None


def _walk(node):
    """Depth-first pre-order traversal of a tree-sitter subtree."""
    yield node
    for child in node.children:
        yield from _walk(child)


def collect_declaration_spans(tree) -> list[TokenSpan]:
    """Collect byte spans of invariant declaration vertices in ``tree``.

    Includes (a) every node whose type is in :data:`DECLARATION_NODE_TYPES` and
    (b) the top-level global-scope children of the root node (module/program).
    """
    spans: list[TokenSpan] = []
    root = tree.root_node
    for node in _walk(root):
        if node.type in DECLARATION_NODE_TYPES:
            spans.append((node.start_byte, node.end_byte))
    for child in root.children:
        if child.type in GLOBAL_SCOPE_NODE_TYPES:
            spans.append((child.start_byte, child.end_byte))
    return spans


class ASTSlicer:
    """AST-aware semantic slicer producing permanent page masks.

    Parameters
    ----------
    language:
        tree-sitter language name (default ``"python"``).
    block_size:
        Tokens per physical page :math:`B`.
    pin_ratio:
        Cap ratio :math:`\\gamma` bounding the pinned-page budget.
    tokenizer:
        Callable mapping source text to byte spans.  Defaults to
        :func:`default_tokenize`; pass a HuggingFace tokenizer wrapper for
        production alignment with the model vocabulary.
    parser:
        Optional pre-built tree-sitter parser.  When omitted the slicer attempts
        to build one automatically and degrades to an empty mask on failure.
    """

    def __init__(
        self,
        language: str = "python",
        block_size: int = 16,
        pin_ratio: float = 0.30,
        tokenizer: Optional[Callable[[str], Sequence[TokenSpan]]] = None,
        parser=None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not 0.0 <= pin_ratio <= 1.0:
            raise ValueError(f"pin_ratio must be in [0, 1], got {pin_ratio}")

        self.language = language
        self.block_size = block_size
        self.pin_ratio = pin_ratio
        self._tokenizer = tokenizer if tokenizer is not None else default_tokenize
        self._parser = parser if parser is not None else _try_build_parser(language)

    @property
    def parser_available(self) -> bool:
        """``True`` iff a tree-sitter parser is available for AST slicing."""
        return self._parser is not None

    def _declaration_spans(self, code: str) -> list[TokenSpan]:
        if not code or self._parser is None:
            return []
        tree = self._parser.parse(code.encode("utf-8"))
        return collect_declaration_spans(tree)

    def tokenize(self, code: str) -> list[TokenSpan]:
        """Return the token byte spans for ``code`` using the configured tokenizer."""
        return list(self._tokenizer(code))

    def token_count(self, code: str) -> int:
        """Return the number of tokens produced for ``code``."""
        return len(self._tokenizer(code))

    def invariant_token_mask(self, code: str) -> npt.NDArray[np.bool_]:
        """Return the token-level invariant mask :math:`\\mathcal{T}_{\\text{AST}}`."""
        token_spans = self._tokenizer(code)
        declarations = self._declaration_spans(code)
        if not declarations:
            return np.zeros(len(token_spans), dtype=np.bool_)
        return invariant_token_mask(token_spans, declarations)

    def page_mask(
        self,
        code: str,
        num_blocks: Optional[int] = None,
        pin_cap: Optional[int] = None,
    ) -> PageMask:
        """Return the capped permanent page mask for ``code``.

        Parameters
        ----------
        num_blocks:
            Physical capacity :math:`N` (the number of pages the prompt occupies).
            Defaults to ``ceil(T / block_size)``.
        pin_cap:
            Absolute pinned-page cap :math:`\\mathcal{B}_{\\text{pinned}}` (e.g.
            ``floor(gamma * B_total)`` where :math:`B_{total}` is the retention
            budget).  Defaults to ``floor(gamma * num_blocks)``.
        """
        token_spans = self._tokenizer(code)
        declarations = self._declaration_spans(code)
        if declarations:
            token_mask = invariant_token_mask(token_spans, declarations)
        else:
            token_mask = np.zeros(len(token_spans), dtype=np.bool_)

        page_mask, capped = page_mask_from_tokens(
            token_mask,
            self.block_size,
            num_blocks=num_blocks,
            pin_ratio=self.pin_ratio,
            pin_cap=pin_cap,
        )
        pinned = np.flatnonzero(page_mask).astype(np.int64)
        if pin_cap is None:
            effective_cap = int(np.floor(self.pin_ratio * int(page_mask.shape[0])))
        else:
            effective_cap = int(pin_cap)
        return PageMask(
            token_mask=token_mask,
            page_mask=page_mask,
            pinned_block_indices=pinned,
            block_size=self.block_size,
            num_tokens=len(token_spans),
            num_blocks=int(page_mask.shape[0]),
            pin_ratio=self.pin_ratio,
            pin_cap=effective_cap,
            capped=capped,
            degraded=not declarations,
        )

    def __call__(
        self,
        code: str,
        num_blocks: Optional[int] = None,
        pin_cap: Optional[int] = None,
    ) -> PageMask:
        """Alias for :meth:`page_mask`."""
        return self.page_mask(code, num_blocks=num_blocks, pin_cap=pin_cap)


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
]
