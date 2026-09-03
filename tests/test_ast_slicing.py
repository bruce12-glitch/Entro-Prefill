"""Tests for Scenario 1: AST-aware semantic slicing."""

from __future__ import annotations

import numpy as np
import pytest

from page_entrokv.ast_slicing import (
    ASTSlicer,
    DECLARATION_NODE_TYPES,
    default_tokenize,
    invariant_token_mask,
    page_mask_from_tokens,
)

PY_CODE = """\
import os
from typing import List

GLOBAL_LIMIT = 128

def add(a: int, b: int) -> int:
    return a + b

class Counter:
    total: int = 0
"""

NATURAL_LANGUAGE = (
    "The quick brown fox jumps over the lazy dog. It then kept running "
    "across the field without stopping for anything."
)


def test_expected_node_types_are_declared() -> None:
    for node_type in (
        "class_definition",
        "function_definition",
        "type_annotation",
        "import_statement",
    ):
        assert node_type in DECLARATION_NODE_TYPES


def test_default_tokenize_is_word_level() -> None:
    spans = default_tokenize("def f(x):\n    return x")
    assert len(spans) > 5
    text = "def f(x):\n    return x"
    assert all(0 <= s < e <= len(text) for s, e in spans)


def test_invariant_token_mask_projects_declarations() -> None:
    spans = default_tokenize(PY_CODE)
    # A declaration span covering the `def` keyword line.
    decl = [(PY_CODE.index("def add"), PY_CODE.index("return a + b"))]
    mask = invariant_token_mask(spans, decl)
    assert mask.shape == (len(spans),)
    assert bool(mask.any())
    # Tokens before the declaration must be untouched.
    first_decl_token = next(i for i, m in enumerate(mask) if m)
    assert not mask[:first_decl_token].any()


def test_invariant_token_mask_empty_inputs() -> None:
    assert invariant_token_mask([], []).size == 0
    assert not invariant_token_mask([(0, 1)], []).any()


def test_invariant_token_mask_merges_overlapping_spans() -> None:
    spans = [(0, 1), (1, 2), (2, 3), (3, 4)]
    decl = [(0, 2), (1, 3)]
    mask = invariant_token_mask(spans, decl)
    assert mask.tolist() == [True, True, True, False]


def test_page_mask_from_tokens_basic() -> None:
    mask = np.array([True, False, False, False, True, False], dtype=bool)
    page_mask, capped = page_mask_from_tokens(mask, block_size=2, pin_ratio=1.0)
    # Blocks: [T,F] -> pinned, [F,F] -> not, [T,F] -> pinned.
    assert page_mask.tolist() == [True, False, True]
    assert not capped


def test_page_mask_applies_pin_ratio_cap() -> None:
    # 10 pages all pinned, cap ratio 0.3 -> at most 3 pages.
    mask = np.ones(10, dtype=bool)
    page_mask, capped = page_mask_from_tokens(mask, block_size=1, pin_ratio=0.3)
    assert int(page_mask.sum()) == 3
    assert capped


def test_page_mask_density_keeps_densest_pages() -> None:
    mask = np.array([True, True, True, False, False, False], dtype=bool)
    page_mask, capped = page_mask_from_tokens(mask, block_size=2, pin_ratio=0.5)
    # Blocks: [T,T] density 2, [T,F] density 1, [F,F] density 0 -> keep block 0.
    assert page_mask.tolist() == [True, False, False]
    assert capped


def test_page_mask_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        page_mask_from_tokens(np.array([True]), block_size=0)
    with pytest.raises(ValueError):
        page_mask_from_tokens(np.array([True]), block_size=1, pin_ratio=1.5)


def test_slicer_degrades_on_natural_language() -> None:
    slicer = ASTSlicer(block_size=8, pin_ratio=0.3)
    result = slicer.page_mask(NATURAL_LANGUAGE)
    assert result.degraded
    assert not result.page_mask.any()
    assert result.pinned_block_indices.size == 0
    assert result.num_blocks > 0


def test_slicer_empty_prompt() -> None:
    slicer = ASTSlicer(block_size=8, pin_ratio=0.3)
    result = slicer.page_mask("")
    assert result.degraded
    assert result.num_tokens == 0


def test_slicer_pins_declarations_when_parser_available() -> None:
    slicer = ASTSlicer(block_size=8, pin_ratio=1.0)
    if not slicer.parser_available:
        pytest.skip("tree-sitter parser unavailable in this environment")
    result = slicer.page_mask(PY_CODE)
    assert result.num_tokens > 0
    assert result.num_blocks == int(np.ceil(result.num_tokens / 8))
    assert result.pinned_block_indices.size > 0
    assert not result.degraded
    # Every pinned block must actually contain at least one invariant token.
    for block in result.pinned_block_indices:
        start = int(block) * 8
        end = min(start + 8, result.num_tokens)
        assert bool(result.token_mask[start:end].any())


def test_slicer_respects_pin_ratio() -> None:
    slicer = ASTSlicer(block_size=4, pin_ratio=0.25)
    if not slicer.parser_available:
        pytest.skip("tree-sitter parser unavailable in this environment")
    result = slicer.page_mask(PY_CODE)
    max_pinned = int(np.floor(0.25 * result.num_blocks))
    assert result.pinned_block_indices.size <= max_pinned
