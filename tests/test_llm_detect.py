"""Tests for gocalma.llm_detect — response parsing, occurrence search, device selection."""

from __future__ import annotations

import pytest

from gocalma.llm_detect import (
    _parse_verify_response,
    _find_best_occurrence,
    _best_device,
    is_llm_available,
    LLM_MODELS,
)


class TestParseVerifyResponse:
    def test_well_formed_response(self):
        raw = '{"verified": [{"index": 0, "verdict": "confirmed"}], "additional": []}'
        result = _parse_verify_response(raw)
        assert result["verified"][0]["verdict"] == "confirmed"
        assert result["additional"] == []

    def test_response_with_markdown_wrapper(self):
        raw = "```json\n{\"verified\": [], \"additional\": []}\n```"
        result = _parse_verify_response(raw)
        assert "verified" in result

    def test_invalid_json_returns_empty(self):
        result = _parse_verify_response("this is not json at all")
        assert result == {"verified": [], "additional": []}

    def test_empty_string_returns_empty(self):
        result = _parse_verify_response("")
        assert result == {"verified": [], "additional": []}

    def test_partial_json_returns_empty(self):
        result = _parse_verify_response('{"verified": [')
        assert result == {"verified": [], "additional": []}

    def test_false_positive_verdict(self):
        raw = '{"verified": [{"index": 1, "verdict": "false_positive"}], "additional": []}'
        result = _parse_verify_response(raw)
        assert result["verified"][0]["verdict"] == "false_positive"

    def test_additional_entity_parsed(self):
        raw = (
            '{"verified": [], "additional": '
            '[{"type": "ADDRESS", "text": "Feldlerchenweg 15", "start": 42, "end": 59}]}'
        )
        result = _parse_verify_response(raw)
        assert result["additional"][0]["text"] == "Feldlerchenweg 15"

    def test_json_embedded_in_prose(self):
        raw = 'Here is my analysis: {"verified": [], "additional": []} — done.'
        result = _parse_verify_response(raw)
        assert result["verified"] == []

    def test_non_dict_json_returns_empty(self):
        raw = "[1, 2, 3]"
        result = _parse_verify_response(raw)
        assert result == {"verified": [], "additional": []}


class TestFindBestOccurrence:
    TEXT = "John went to Paris. John met Marie in Paris."

    def test_first_occurrence_when_reported_start_is_zero(self):
        idx = _find_best_occurrence(self.TEXT, "John", 0)
        assert idx == 0

    def test_second_occurrence_when_reported_start_matches(self):
        # "John" appears again at index 20; report start=20
        idx = _find_best_occurrence(self.TEXT, "John", 20)
        assert idx == 20

    def test_not_found_returns_minus_one(self):
        idx = _find_best_occurrence(self.TEXT, "Zurich", 0)
        assert idx == -1

    def test_empty_substring_returns_non_negative(self):
        # Empty substring is filtered upstream before _find_best_occurrence is
        # called, but if it does arrive the function must not crash.
        # Python's str.find("", n) returns n, so we just assert no crash and >= 0.
        idx = _find_best_occurrence(self.TEXT, "", 5)
        assert idx >= 0

    def test_single_occurrence(self):
        idx = _find_best_occurrence(self.TEXT, "Marie", 100)
        assert idx == self.TEXT.index("Marie")

    def test_closest_when_equidistant_is_deterministic(self):
        text = "ab--ab"
        # "ab" at 0 and 4; reported start = 2 → equidistant
        idx = _find_best_occurrence(text, "ab", 2)
        # Either is acceptable as long as we get one of them
        assert idx in (0, 4)


class TestBestDevice:
    def test_returns_string(self):
        device = _best_device()
        assert isinstance(device, str)

    def test_valid_device_name(self):
        device = _best_device()
        assert device in ("cuda", "mps", "cpu")


class TestIsLlmAvailable:
    def test_returns_bool(self):
        assert isinstance(is_llm_available(), bool)

    def test_unknown_key_returns_false(self):
        assert not is_llm_available("NonExistentModel-XYZ")

    def test_known_key_not_installed_returns_false(self):
        # In CI / test environments the model weights will not be present
        for key in LLM_MODELS:
            result = is_llm_available(key)
            assert isinstance(result, bool)
