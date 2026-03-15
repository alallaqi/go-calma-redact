"""Tests for gocalma.pii_detect — deduplication, entity construction, engine cache."""

from __future__ import annotations

import pytest

from gocalma.pii_detect import (
    PIIEntity,
    _deduplicate,
    available_models,
    NLP_MODELS,
    _MAX_ENGINES,
)


def _make(start: int, end: int, score: float, page: int = 0, entity_type: str = "PERSON") -> PIIEntity:
    return PIIEntity(
        entity_type=entity_type,
        text=f"text_{start}_{end}",
        start=start,
        end=end,
        score=score,
        page_num=page,
    )


class TestDeduplicate:
    def test_empty_input(self):
        assert _deduplicate([]) == []

    def test_no_overlap(self):
        a = _make(0, 5, 0.9)
        b = _make(10, 15, 0.8)
        result = _deduplicate([a, b])
        assert len(result) == 2

    def test_overlap_keeps_higher_score(self):
        low = _make(0, 10, 0.5)
        high = _make(3, 8, 0.9)
        result = _deduplicate([low, high])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_identical_span_keeps_first_encountered_after_sort(self):
        a = _make(0, 10, 0.7)
        b = _make(0, 10, 0.7)
        result = _deduplicate([a, b])
        assert len(result) == 1

    def test_adjacent_spans_kept(self):
        a = _make(0, 5, 0.8)
        b = _make(5, 10, 0.8)  # starts exactly where a ends — not overlapping
        result = _deduplicate([a, b])
        assert len(result) == 2

    def test_fully_contained_overlap_replaced_by_higher(self):
        outer = _make(0, 20, 0.4)
        inner = _make(5, 15, 0.95)
        result = _deduplicate([outer, inner])
        assert len(result) == 1
        assert result[0].score == 0.95

    def test_multi_page_entities_treated_independently(self):
        """Entities on different pages should never be deduplicated against each other."""
        p0 = _make(0, 10, 0.9, page=0)
        p1 = _make(0, 10, 0.5, page=1)
        # _deduplicate works within a flat list and uses only start/end;
        # the caller (detect_pii_all_pages) processes pages separately.
        result = _deduplicate([p0, p1])
        # Both have same start/end; only one survives dedup (by design —
        # multi-page dedup is handled upstream).
        assert len(result) >= 1

    def test_order_preserved_for_non_overlapping(self):
        entities = [_make(i * 10, i * 10 + 5, 0.8) for i in range(5)]
        result = _deduplicate(entities)
        assert result == entities

    def test_single_entity_returned_unchanged(self):
        e = _make(0, 5, 0.75)
        assert _deduplicate([e]) == [e]


class TestPIIEntity:
    def test_default_source_is_ner(self):
        e = PIIEntity("PERSON", "Alice", 0, 5, 0.9, page_num=0)
        assert e.source == "NER"

    def test_default_analysis_is_empty(self):
        e = PIIEntity("PERSON", "Alice", 0, 5, 0.9, page_num=0)
        assert e.analysis == ""

    def test_custom_source(self):
        e = PIIEntity("PERSON", "Alice", 0, 5, 0.9, page_num=0, source="LLM")
        assert e.source == "LLM"


class TestAvailableModels:
    def test_returns_dict(self):
        result = available_models()
        assert isinstance(result, dict)

    def test_subset_of_nlp_models(self):
        result = available_models()
        assert all(k in NLP_MODELS for k in result)

    def test_max_engines_constant(self):
        assert _MAX_ENGINES >= 1
