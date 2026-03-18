"""Tests for gocalma.redactor — replacement text, HMAC hash, redact_pdf, page ops."""

from __future__ import annotations

import os

import fitz
import pytest

from gocalma.pii_detect import PIIEntity
from gocalma.redactor import (
    _replacement_text,
    APPROACHES,
    DEFAULT_APPROACH,
    redact_pdf,
    page_count,
    render_page,
    extract_words,
    _flatten_page_as_image,
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(text: str = "John Smith lives at Musterstrasse 1") -> bytes:
    """Create a single-page PDF with selectable text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _make_entity(text: str = "John Smith", start: int = 0, page_num: int = 0) -> PIIEntity:
    return PIIEntity(
        entity_type="PERSON",
        text=text,
        start=start,
        end=start + len(text),
        score=0.9,
        page_num=page_num,
    )


# ---------------------------------------------------------------------------
# HMAC hash security
# ---------------------------------------------------------------------------

class TestHmacHash:
    def test_hmac_differs_from_plain_hash(self):
        ent = _entity()
        hmac_key = os.urandom(32)
        plain = _replacement_text(ent, "hash")
        salted = _replacement_text(ent, "hash", hmac_key=hmac_key)
        assert plain != salted

    def test_hmac_deterministic_with_same_key(self):
        ent = _entity()
        key = os.urandom(32)
        a = _replacement_text(ent, "hash", hmac_key=key)
        b = _replacement_text(ent, "hash", hmac_key=key)
        assert a == b

    def test_hmac_differs_with_different_keys(self):
        ent = _entity()
        a = _replacement_text(ent, "hash", hmac_key=os.urandom(32))
        b = _replacement_text(ent, "hash", hmac_key=os.urandom(32))
        assert a != b

    def test_hmac_format_matches_plain(self):
        ent = _entity()
        result = _replacement_text(ent, "hash", hmac_key=os.urandom(32))
        assert result.startswith("[#")
        assert result.endswith("]")
        inner = result[2:-1]
        assert len(inner) == 12
        assert all(c in "0123456789abcdef" for c in inner)

    def test_hmac_short_pii_not_trivially_reversible(self):
        """Different runs with different keys produce different hashes for same input."""
        ent = _entity(text="044 123 45 67")  # Swiss phone number
        hashes = {_replacement_text(ent, "hash", hmac_key=os.urandom(32)) for _ in range(10)}
        assert len(hashes) == 10  # All different


# ---------------------------------------------------------------------------
# redact_pdf — end-to-end
# ---------------------------------------------------------------------------

class TestRedactPdf:
    def test_returns_bytes_and_mapping(self):
        pdf = _make_pdf()
        ent = _make_entity()
        result_bytes, mapping = redact_pdf(pdf, [ent], approach="replace")
        assert isinstance(result_bytes, bytes)
        assert isinstance(mapping, dict)
        assert len(mapping) >= 1

    def test_mapping_contains_original_text(self):
        pdf = _make_pdf()
        ent = _make_entity()
        _, mapping = redact_pdf(pdf, [ent], approach="replace")
        assert "John Smith" in mapping.values()

    def test_mapping_labels_have_entity_type(self):
        pdf = _make_pdf()
        ent = _make_entity()
        _, mapping = redact_pdf(pdf, [ent], approach="replace")
        for label in mapping:
            if label.startswith("__"):
                continue  # skip metadata keys like __hmac_key__
            assert "PERSON" in label

    def test_hash_approach_stores_hmac_key(self):
        pdf = _make_pdf()
        ent = _make_entity()
        _, mapping = redact_pdf(pdf, [ent], approach="hash")
        assert "__hmac_key__" in mapping
        # Key should be 64 hex chars (32 bytes)
        assert len(mapping["__hmac_key__"]) == 64

    def test_non_hash_approach_has_no_hmac_key(self):
        pdf = _make_pdf()
        ent = _make_entity()
        _, mapping = redact_pdf(pdf, [ent], approach="replace")
        assert "__hmac_key__" not in mapping

    def test_empty_entities_returns_unchanged(self):
        pdf = _make_pdf()
        result_bytes, mapping = redact_pdf(pdf, [], approach="replace")
        assert isinstance(result_bytes, bytes)
        assert mapping == {}

    def test_flatten_mode_produces_valid_pdf(self):
        pdf = _make_pdf()
        ent = _make_entity()
        result_bytes, _ = redact_pdf(pdf, [ent], approach="redact", flatten=True)
        # Should be openable
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        assert len(doc) == 1
        doc.close()

    def test_reversible_mode_produces_valid_pdf(self):
        pdf = _make_pdf()
        ent = _make_entity()
        result_bytes, _ = redact_pdf(pdf, [ent], approach="replace", flatten=False)
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        assert len(doc) == 1
        doc.close()

    def test_reversible_annotations_do_not_contain_original_text(self):
        """Annotation metadata must not leak the original PII text."""
        pdf = _make_pdf("Secret Name at home")
        ent = PIIEntity("PERSON", "Secret Name", 0, 11, 0.9, page_num=0)
        result_bytes, _ = redact_pdf(pdf, [ent], approach="replace", flatten=False)
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        page = doc[0]
        for annot in page.annots():
            info = annot.info
            # The content field should be the tracking label, never the original text
            assert "Secret Name" not in info.get("content", "")
        doc.close()


# ---------------------------------------------------------------------------
# page_count / render_page / extract_words
# ---------------------------------------------------------------------------

class TestPageOps:
    def test_page_count_single(self):
        pdf = _make_pdf()
        assert page_count(pdf) == 1

    def test_page_count_multi(self):
        doc = fitz.open()
        for _ in range(5):
            doc.new_page()
        data = doc.tobytes()
        doc.close()
        assert page_count(data) == 5

    def test_render_page_returns_png(self):
        pdf = _make_pdf()
        png = render_page(pdf, 0)
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"

    def test_extract_words_returns_list(self):
        pdf = _make_pdf("Hello World")
        words = extract_words(pdf, 0)
        assert isinstance(words, list)
        assert len(words) >= 2
        texts = [w["text"] for w in words]
        assert "Hello" in texts
        assert "World" in texts

    def test_extract_words_have_coordinates(self):
        pdf = _make_pdf("Test")
        words = extract_words(pdf, 0)
        assert len(words) >= 1
        w = words[0]
        for key in ("x_pct", "y_pct", "w_pct", "h_pct", "x0", "y0", "x1", "y1"):
            assert key in w


# ---------------------------------------------------------------------------
# _flatten_page_as_image
# ---------------------------------------------------------------------------

class TestFlattenPage:
    def test_flattened_page_is_image_only(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "Sensitive text here", fontsize=12)
        _flatten_page_as_image(doc, 0, dpi=72)
        # After flattening, the page should have no selectable text
        page = doc[0]
        text = page.get_text("text").strip()
        assert text == ""
        # But should still have content (the image)
        assert len(page.get_images()) >= 1
        doc.close()

    def test_page_dimensions_preserved(self):
        doc = fitz.open()
        doc.new_page(width=400, height=600)
        _flatten_page_as_image(doc, 0, dpi=72)
        page = doc[0]
        assert abs(page.rect.width - 400) < 1
        assert abs(page.rect.height - 600) < 1
        doc.close()
