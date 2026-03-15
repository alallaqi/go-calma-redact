"""Tests for gocalma.redactor — replacement text logic and approach coverage."""

from __future__ import annotations

import pytest

from gocalma.pii_detect import PIIEntity
from gocalma.redactor import _replacement_text, APPROACHES, DEFAULT_APPROACH


def _entity(entity_type: str = "PERSON", text: str = "John Smith") -> PIIEntity:
    return PIIEntity(
        entity_type=entity_type,
        text=text,
        start=0,
        end=len(text),
        score=0.9,
        page_num=0,
    )


class TestReplacementText:
    def test_redact_returns_empty(self):
        assert _replacement_text(_entity(), "redact") == ""

    def test_replace_returns_type_label(self):
        result = _replacement_text(_entity("EMAIL_ADDRESS"), "replace")
        assert result == "<EMAIL_ADDRESS>"

    def test_mask_returns_asterisks_matching_length(self):
        ent = _entity(text="Alice")
        result = _replacement_text(ent, "mask")
        assert result == "*****"
        assert len(result) == len(ent.text)

    def test_hash_returns_prefixed_hex(self):
        result = _replacement_text(_entity(), "hash")
        assert result.startswith("[#")
        assert result.endswith("]")
        # 12 hex chars between [# and ]
        inner = result[2:-1]
        assert len(inner) == 12
        assert all(c in "0123456789abcdef" for c in inner)

    def test_hash_is_deterministic(self):
        ent = _entity()
        assert _replacement_text(ent, "hash") == _replacement_text(ent, "hash")

    def test_hash_differs_for_different_text(self):
        a = _replacement_text(_entity(text="Alice"), "hash")
        b = _replacement_text(_entity(text="Bob"), "hash")
        assert a != b

    def test_encrypt_embeds_label(self):
        result = _replacement_text(_entity(), "encrypt", label="[PERSON_abc123]")
        assert result.startswith("[enc:")
        assert "PERSON" in result or "abc123" in result[:20]

    def test_encrypt_without_label_still_returns_something(self):
        result = _replacement_text(_entity(), "encrypt", label="")
        assert result.startswith("[enc:")

    def test_highlight_returns_original_text(self):
        ent = _entity(text="Jane Doe")
        assert _replacement_text(ent, "highlight") == "Jane Doe"

    def test_synthesize_known_type(self):
        ent = _entity(entity_type="PERSON")
        assert _replacement_text(ent, "synthesize") == "John Doe"

    def test_synthesize_unknown_type_falls_back(self):
        ent = _entity(entity_type="CUSTOM_TYPE_XYZ")
        result = _replacement_text(ent, "synthesize")
        assert result == "<REDACTED>"

    def test_unknown_approach_falls_back_to_replace(self):
        result = _replacement_text(_entity("LOCATION"), "nonexistent_approach")
        assert result == "<LOCATION>"


class TestApproaches:
    def test_all_approaches_covered(self):
        ent = _entity()
        for approach in APPROACHES:
            result = _replacement_text(ent, approach)
            assert isinstance(result, str)

    def test_default_approach_in_approaches(self):
        assert DEFAULT_APPROACH in APPROACHES

    def test_approaches_is_non_empty(self):
        assert len(APPROACHES) > 0
