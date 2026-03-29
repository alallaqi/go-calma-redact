"""Regression tests for entity recognition — deduplication and filtering."""

from __future__ import annotations

from gocalma.regex_patterns import run_regex, merge_with_priority, filter_implausible


def test_no_duplicate_police_number():
    """Same entity detected by regex + NER + LLM should appear exactly once."""
    text = "Polizeinummer: 756.1234.5678.90"
    regex_result = run_regex(text)
    ner_result = [{"text": "756.1234.5678.90", "start": 15,
                   "end": 31, "type": "CH_AHV", "priority": 10}]
    llm_result = [{"text": "756.1234.5678.90", "start": 15,
                   "end": 31, "type": "CH_AHV", "priority": 10}]
    merged = merge_with_priority(regex_result, ner_result, llm_result)
    matches = [e for e in merged if "756.1234.5678.90" in e["text"]]
    assert len(matches) == 1, f"Expected 1, got {len(matches)}"


def test_generic_street_not_flagged():
    """A bare street name with no house number should not be flagged as PII."""
    from gocalma.llm_detect import is_generic_street_name
    assert is_generic_street_name("Bahnhofstrasse") == True
    assert is_generic_street_name("Hauptstrasse") == True
    assert is_generic_street_name("14 Bahnhofstrasse") == False
    assert is_generic_street_name("Bahnhofstrasse 14") == False


# ---------------------------------------------------------------------------
# Fix 1 — Minimum entity length gate
# ---------------------------------------------------------------------------

def test_single_letter_filtered():
    entities = [
        {"text": "A", "type": "ORG", "priority": 4},
        {"text": "GI", "type": "ORG", "priority": 4},
        {"text": "Charles Muster", "type": "PERSON", "priority": 6},
    ]
    result = filter_implausible(entities)
    assert len(result) == 1
    assert result[0]["text"] == "Charles Muster"


def test_whitespace_only_filtered():
    entities = [{"text": "   ", "type": "PERSON", "priority": 6}]
    assert filter_implausible(entities) == []


def test_punctuation_only_filtered():
    entities = [{"text": "---", "type": "UNKNOWN", "priority": 3}]
    assert filter_implausible(entities) == []


def test_valid_three_char_entity_kept():
    entities = [{"text": "Bob", "type": "PERSON", "priority": 6}]
    assert len(filter_implausible(entities)) == 1


# ---------------------------------------------------------------------------
# Fix 2 — Context-aware location filter
# ---------------------------------------------------------------------------

def test_switzerland_in_product_description_not_flagged():
    from gocalma.pii_detect import is_contextual_false_positive
    entity = {"text": "Switzerland", "type": "LOCATION",
              "start": 50, "end": 61}
    context = "x" * 20 + "General ward, anywhere in " + "Switzerland" + " more text"
    # Adjust start/end to match actual position in context
    start = context.index("Switzerland")
    entity["start"] = start
    entity["end"] = start + len("Switzerland")
    assert is_contextual_false_positive(entity, context) == True


def test_switzerland_in_address_IS_flagged():
    from gocalma.pii_detect import is_contextual_false_positive
    context = "Feldlerchenweg 15, 3360 Herzogenbuchsee, Switzerland"
    start = context.index("Switzerland")
    entity = {"text": "Switzerland", "type": "LOCATION",
              "start": start, "end": start + 11}
    assert is_contextual_false_positive(entity, context) == False


def test_premium_region_not_flagged():
    from gocalma.pii_detect import is_contextual_false_positive
    context = "Premium region BE 3"
    start = context.index("BE 3")
    entity = {"text": "BE 3", "type": "LOCATION",
              "start": start, "end": start + 4}
    assert is_contextual_false_positive(entity, context) == True


def test_abbreviation_table_not_flagged():
    from gocalma.pii_detect import is_contextual_false_positive
    context = "GIC General Insurance Conditions AIC Additional Insurance"
    entity = {"text": "GIC", "type": "ORG",
              "start": 0, "end": 3}
    assert is_contextual_false_positive(entity, context) == True


def test_abbreviation_not_flagged_without_context():
    """An abbreviation NOT near legend keywords should be kept."""
    from gocalma.pii_detect import is_contextual_false_positive
    context = "The GIC called yesterday about the claim."
    start = context.index("GIC")
    entity = {"text": "GIC", "type": "ORG",
              "start": start, "end": start + 3}
    assert is_contextual_false_positive(entity, context) == False


# ---------------------------------------------------------------------------
# Fix 3 — Insurance number pattern
# ---------------------------------------------------------------------------

def test_insurance_number_grouped_format_matched():
    """The explicit 3-3-3 insurance number format should be detected."""
    hits = run_regex("Insurance no. 100 452 956")
    types = [h["type"] for h in hits]
    assert "INSURANCE_NUMBER" in types


def test_short_code_not_matched_as_insurance():
    """Short codes like 'BE 3' should NOT match insurance patterns."""
    hits = run_regex("Premium region BE 3")
    insurance_hits = [h for h in hits if h["type"] == "INSURANCE_NUMBER"]
    assert len(insurance_hits) == 0


# ---------------------------------------------------------------------------
# Fix 4 — Confidence score computation
# ---------------------------------------------------------------------------

def test_regex_entity_always_scores_1():
    from gocalma.pii_detect import compute_confidence
    entity = {
        "text": "756.1234.5678.90", "type": "CH_AHV",
        "source": "regex", "score": 0.5, "start": 0, "end": 16,
    }
    assert compute_confidence(entity, "AHV: 756.1234.5678.90") == 1.0


def test_person_name_gets_floor_score():
    from gocalma.pii_detect import compute_confidence
    entity = {
        "text": "Charles Muster", "type": "PERSON",
        "source": "ner", "score": 0.20, "start": 10, "end": 24,
    }
    # Even with raw score of 0.20, floor + length + context
    # should bring it well above 0.80
    result = compute_confidence(
        entity,
        "xxxxxxxxx Mr Charles Muster, insurance no. 100 452 956",
    )
    assert result >= 0.80


def test_repeated_name_scores_higher():
    from gocalma.pii_detect import compute_confidence
    entity = {
        "text": "Charles Muster", "type": "PERSON",
        "source": "ner", "score": 0.50, "start": 0, "end": 14,
    }
    repeated_doc = "Charles Muster " * 5
    result = compute_confidence(entity, repeated_doc)
    # Repetition boost should apply
    assert result >= 0.90


def test_person_cannot_be_disputed_by_llm():
    """apply_llm_verdict must override LLM false_positive for PERSON type."""
    from gocalma.llm_detect import apply_llm_verdict
    entity = {"text": "Max Muster", "type": "PERSON", "source": "ner"}
    result = apply_llm_verdict(entity, "FALSE_POSITIVE", "seems generic")
    assert result["llm_status"] == "confirmed"


def test_regex_entity_cannot_be_disputed():
    """apply_llm_verdict must override LLM false_positive for regex-sourced entities."""
    from gocalma.llm_detect import apply_llm_verdict
    entity = {"text": "756.1234.5678.90", "type": "CH_AHV", "source": "regex"}
    result = apply_llm_verdict(entity, "FALSE_POSITIVE", "not PII")
    assert result["llm_status"] == "confirmed"
