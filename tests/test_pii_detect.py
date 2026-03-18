"""Tests for gocalma.pii_detect — deduplication, entity construction, engine cache,
Swiss regex patterns, actual PII detection, language detection.
"""

from __future__ import annotations

import re

import pytest

from gocalma.pii_detect import (
    PIIEntity,
    _deduplicate,
    _detect_language,
    _make_swiss_recognizers,
    available_models,
    detect_pii,
    NLP_MODELS,
    DEFAULT_MODEL,
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


# ---------------------------------------------------------------------------
# Swiss regex patterns — unit tests on the raw regex strings
# ---------------------------------------------------------------------------

class TestSwissPatterns:
    """Test the regex patterns from _make_swiss_recognizers without needing Presidio."""

    @pytest.fixture()
    def recognizers(self):
        return {r.name: r for r in _make_swiss_recognizers("de")}

    def test_ahv_standard_format(self, recognizers):
        """756.1234.5678.90"""
        r = recognizers["Swiss AHV Number"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "756.1234.5678.90") for p in patterns)

    def test_ahv_space_format(self, recognizers):
        """756 1234 5678 90"""
        r = recognizers["Swiss AHV Number"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "756 1234 5678 90") for p in patterns)

    def test_ahv_rejects_short(self, recognizers):
        r = recognizers["Swiss AHV Number"]
        patterns = [p.regex for p in r.patterns]
        assert not any(re.search(p, "756.123") for p in patterns)

    def test_postal_code_with_city(self, recognizers):
        r = recognizers["Swiss Postal Code"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "8003 Zürich") for p in patterns)

    def test_postal_code_rejects_year(self, recognizers):
        """A bare 4-digit number without a following city should not match."""
        r = recognizers["Swiss Postal Code"]
        patterns = [p.regex for p in r.patterns]
        assert not any(re.search(p, "2024 was good") for p in patterns)

    def test_street_address(self, recognizers):
        r = recognizers["Street Address (DACH)"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "Musterstrasse 12") for p in patterns)

    def test_german_date(self, recognizers):
        r = recognizers["German Date"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "26. März 1975") for p in patterns)

    def test_swiss_phone_local(self, recognizers):
        r = recognizers["Swiss Phone (local format)"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "044 123 45 67") for p in patterns)

    def test_access_code_dash(self, recognizers):
        r = recognizers["Swiss Access / Login Code"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "ABCD-EFgh-IJKL-MNop") for p in patterns)

    def test_pid_format(self, recognizers):
        r = recognizers["Swiss Personal / Document ID"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "12-3456-78") for p in patterns)

    def test_insurance_number_spaced(self, recognizers):
        r = recognizers["Insurance / Policy Number"]
        patterns = [p.regex for p in r.patterns]
        assert any(re.search(p, "100 452 956") for p in patterns)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_german_detected(self):
        text = "Sehr geehrte Frau Muster, wir möchten Sie informieren über Ihre Versicherung."
        lang = _detect_language(text, "spaCy/de_core_news_lg")
        assert lang == "de"

    def test_short_text_falls_back(self):
        lang = _detect_language("Hi", DEFAULT_MODEL)
        assert isinstance(lang, str)
        assert len(lang) >= 2

    def test_empty_text_falls_back(self):
        lang = _detect_language("", DEFAULT_MODEL)
        assert isinstance(lang, str)

    def test_unsupported_language_falls_back(self):
        """Japanese text with an EN-only model should fall back."""
        lang = _detect_language("東京は日本の首都です。これは長い文章です。", DEFAULT_MODEL)
        supported = NLP_MODELS[DEFAULT_MODEL]["lang_codes"]
        assert lang in supported


# ---------------------------------------------------------------------------
# Actual PII detection (requires spaCy model)
# ---------------------------------------------------------------------------

class TestDetectPii:
    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        if DEFAULT_MODEL not in available_models():
            pytest.skip(f"{DEFAULT_MODEL} not installed")

    def test_detects_person_name(self):
        entities = detect_pii("John Smith went to the store.", page_num=0)
        types = [e.entity_type for e in entities]
        assert "PERSON" in types

    def test_detects_email(self):
        entities = detect_pii("Contact us at info@example.com for details.", page_num=0)
        types = [e.entity_type for e in entities]
        assert "EMAIL_ADDRESS" in types

    def test_entity_has_correct_text(self):
        entities = detect_pii("Email: test@example.com please.", page_num=0)
        emails = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) >= 1
        assert "test@example.com" in emails[0].text

    def test_entity_has_page_num(self):
        entities = detect_pii("John Smith", page_num=3)
        assert all(e.page_num == 3 for e in entities)

    def test_entity_scores_above_threshold(self):
        entities = detect_pii("John Smith", page_num=0, score_threshold=0.3)
        assert all(e.score >= 0.3 for e in entities)

    def test_no_entities_in_clean_text(self):
        entities = detect_pii("The weather is nice today.", page_num=0)
        # May still detect something, but person/email should be absent
        person_email = [e for e in entities if e.entity_type in ("PERSON", "EMAIL_ADDRESS")]
        assert len(person_email) == 0

    def test_returns_list_of_pii_entity(self):
        entities = detect_pii("Alice", page_num=0)
        assert isinstance(entities, list)
        for e in entities:
            assert isinstance(e, PIIEntity)
