"""Answer metrics: EM/F1 lexical scoring (no LLM required)."""

from __future__ import annotations

import pytest

from eval.answer_metrics import exact_match, token_f1


def test_exact_match_normalizes_case_punctuation_articles():
    assert exact_match("The Cat.", "a cat") == 1.0        # -> "cat" == "cat"
    assert exact_match("cats", "cat") == 0.0              # not the same token


def test_token_f1_partial_overlap():
    # pred -> {cat, sat}; gold -> {cat, sat, down}; overlap 2
    f1 = token_f1("the cat sat", "a cat sat down")
    # precision 2/2=1.0, recall 2/3 -> f1 = 2*1*(2/3)/(1+2/3) = 0.8
    assert f1 == pytest.approx(0.8)


def test_token_f1_no_overlap_is_zero():
    assert token_f1("apples oranges", "cars trucks") == 0.0


def test_token_f1_identical_is_one():
    assert token_f1("vector database", "vector database") == pytest.approx(1.0)
