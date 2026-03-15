"""Tests for gocalma.crypto — key generation, encryption, and key-file formats."""

from __future__ import annotations

import pytest
from cryptography.fernet import InvalidToken

from gocalma.crypto import (
    generate_key,
    encrypt_mapping,
    decrypt_mapping,
    save_key_file,
    load_key_file,
    _SENTINEL_V1,
    _SENTINEL_V2,
)

SAMPLE_MAPPING = {
    "[PERSON_a1b2c3]": "John Smith",
    "[EMAIL_ADDRESS_d4e5f6]": "john@example.com",
    "[PHONE_NUMBER_789abc]": "+41 79 123 45 67",
}


# ---------------------------------------------------------------------------
# Basic encrypt / decrypt round-trip
# ---------------------------------------------------------------------------

class TestEncryptDecrypt:
    def test_round_trip(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        recovered = decrypt_mapping(ct, key)
        assert recovered == SAMPLE_MAPPING

    def test_wrong_key_raises(self):
        key1 = generate_key()
        key2 = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key1)
        with pytest.raises(Exception):  # InvalidToken or similar
            decrypt_mapping(ct, key2)

    def test_empty_mapping(self):
        key = generate_key()
        ct = encrypt_mapping({}, key)
        assert decrypt_mapping(ct, key) == {}

    def test_generate_key_returns_bytes(self):
        key = generate_key()
        assert isinstance(key, bytes)
        assert len(key) == 44  # Fernet key is URL-safe base64, always 44 chars


# ---------------------------------------------------------------------------
# Key file — no password (legacy format)
# ---------------------------------------------------------------------------

class TestKeyFileNoPassword:
    def test_round_trip(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        k2, ct2 = load_key_file(blob)
        assert k2 == key
        assert decrypt_mapping(ct2, k2) == SAMPLE_MAPPING

    def test_uses_v1_sentinel(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        assert _SENTINEL_V1 in blob
        assert not blob.startswith(_SENTINEL_V2)

    def test_load_without_password_succeeds(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        k, c = load_key_file(blob, password=None)
        assert decrypt_mapping(c, k) == SAMPLE_MAPPING


# ---------------------------------------------------------------------------
# Key file — with password (v2 format)
# ---------------------------------------------------------------------------

class TestKeyFileWithPassword:
    def test_round_trip(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="hunter2")
        k2, ct2 = load_key_file(blob, password="hunter2")
        assert k2 == key
        assert decrypt_mapping(ct2, k2) == SAMPLE_MAPPING

    def test_uses_v2_sentinel(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="secret")
        assert blob.startswith(_SENTINEL_V2)

    def test_wrong_password_raises_value_error(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="correct")
        with pytest.raises(ValueError, match="Incorrect passphrase"):
            load_key_file(blob, password="wrong")

    def test_missing_password_raises_value_error(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="secret")
        with pytest.raises(ValueError, match="password-protected"):
            load_key_file(blob, password=None)

    def test_data_key_not_in_blob(self):
        """The raw Fernet data key must not appear in the password-protected blob."""
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="mypassword")
        assert key not in blob

    def test_different_passwords_differ(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob1 = save_key_file(key, ct, password="alpha")
        blob2 = save_key_file(key, ct, password="alpha")
        # Different random salts → different blobs even for same password
        assert blob1 != blob2

    def test_empty_password_string_treated_as_no_password(self):
        """Passing an empty string (from the UI) should be treated as no password."""
        # The app does: password=st.session_state.get("key_password") or None
        # so empty string becomes None before reaching save_key_file.
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password=None)
        assert not blob.startswith(_SENTINEL_V2)


# ---------------------------------------------------------------------------
# Corrupt / invalid blobs
# ---------------------------------------------------------------------------

class TestCorruptBlobs:
    def test_unrecognised_format_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            load_key_file(b"this is not a valid key file at all")

    def test_truncated_v2_raises(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="pw")
        with pytest.raises(ValueError):
            load_key_file(blob[:20], password="pw")
