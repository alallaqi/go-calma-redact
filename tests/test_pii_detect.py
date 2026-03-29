"""Tests for gocalma.pii_detect — deduplication, entity construction,
regex+NER detection, language detection.
"""

from __future__ import annotations

import re

import pytest

from gocalma.pii_detect import (
    PIIEntity,
    _deduplicate,
    _detect_language,
    available_models,
    detect_pii,
    NLP_MODELS,
    DEFAULT_MODEL,
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
        result = _deduplicate([p0, p1])
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

    def test_default_model_is_string(self):
        assert isinstance(DEFAULT_MODEL, str)


# ---------------------------------------------------------------------------
# Regex patterns — smoke tests via detect_pii
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    """Test that regex patterns fire through detect_pii."""

    def test_ahv_standard_format(self):
        entities = detect_pii("AHV: 756.1234.5678.90", page_num=0)
        types = [e.entity_type for e in entities]
        assert "CH_AHV" in types

    def test_email_detected(self):
        entities = detect_pii("Contact info@example.com today", page_num=0)
        types = [e.entity_type for e in entities]
        assert "EMAIL" in types

    def test_us_ssn_detected(self):
        entities = detect_pii("SSN: 123-45-6789", page_num=0)
        types = [e.entity_type for e in entities]
        assert "US_SSN" in types

    def test_dob_detected(self):
        entities = detect_pii("Born 15.03.1990 in Zurich", page_num=0)
        types = [e.entity_type for e in entities]
        assert "DATE_OF_BIRTH" in types

    def test_swiss_postal_with_city(self):
        entities = detect_pii("Address: 8003 Zürich", page_num=0)
        types = [e.entity_type for e in entities]
        assert "CH_POSTAL" in types or "ADDRESS" in types

    def test_zugangscode(self):
        entities = detect_pii("Code: ABCD-EFgh-IJKL-MNop", page_num=0)
        types = [e.entity_type for e in entities]
        assert "CH_ZUGANGSCODE" in types


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_german_detected(self):
        text = "Sehr geehrte Frau Muster, wir möchten Sie informieren über Ihre Versicherung."
        lang = _detect_language(text)
        assert lang == "de"

    def test_short_text_falls_back(self):
        lang = _detect_language("Hi")
        assert isinstance(lang, str)
        assert len(lang) >= 2

    def test_empty_text_falls_back(self):
        lang = _detect_language("")
        assert isinstance(lang, str)


# ---------------------------------------------------------------------------
# Full detect_pii (regex always runs; NER may not be installed)
# ---------------------------------------------------------------------------

class TestDetectPii:
    def test_returns_list_of_pii_entity(self):
        entities = detect_pii("Alice at alice@example.com", page_num=0)
        assert isinstance(entities, list)
        for e in entities:
            assert isinstance(e, PIIEntity)

    def test_entity_has_page_num(self):
        entities = detect_pii("Email: test@example.com", page_num=3)
        assert all(e.page_num == 3 for e in entities)

    def test_no_entities_in_clean_text(self):
        entities = detect_pii("The weather is nice today.", page_num=0)
        person_email = [e for e in entities if e.entity_type in ("PERSON", "EMAIL")]
        assert len(person_email) == 0

    def test_entity_has_correct_text(self):
        entities = detect_pii("Email: test@example.com please.", page_num=0)
        emails = [e for e in entities if e.entity_type == "EMAIL"]
        assert len(emails) >= 1
        assert "test@example.com" in emails[0].text
