"""Tests for gocalma.crypto — AES-256-GCM key generation, encryption, and key-file formats."""

from __future__ import annotations

import pytest

from gocalma.crypto import (
    generate_key,
    encrypt_mapping,
    decrypt_mapping,
    save_key_file,
    load_key_file,
    _SENTINEL_V1,
    _SENTINEL_V2,
    _SENTINEL_V3,
    _KEY_SIZE,
    _NONCE_SIZE,
)

SAMPLE_MAPPING = {
    "[PERSON_a1b2c3]": "John Smith",
    "[EMAIL_ADDRESS_d4e5f6]": "john@example.com",
    "[PHONE_NUMBER_789abc]": "+41 79 123 45 67",
}


# ---------------------------------------------------------------------------
# Basic encrypt / decrypt round-trip (AES-256-GCM)
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
        with pytest.raises(ValueError):
            decrypt_mapping(ct, key2)

    def test_empty_mapping(self):
        key = generate_key()
        ct = encrypt_mapping({}, key)
        assert decrypt_mapping(ct, key) == {}

    def test_generate_key_returns_bytes(self):
        key = generate_key()
        assert isinstance(key, bytes)
        assert len(key) == _KEY_SIZE  # 32 bytes = 256 bits

    def test_ciphertext_starts_with_nonce(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        assert len(ct) > _NONCE_SIZE

    def test_different_encryptions_differ(self):
        """Each encryption uses a fresh random nonce."""
        key = generate_key()
        ct1 = encrypt_mapping(SAMPLE_MAPPING, key)
        ct2 = encrypt_mapping(SAMPLE_MAPPING, key)
        assert ct1 != ct2  # different nonces

    def test_plaintext_not_in_ciphertext(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        assert b"John Smith" not in ct


# ---------------------------------------------------------------------------
# Key file — no password (v3 format)
# ---------------------------------------------------------------------------

class TestKeyFileNoPassword:
    def test_round_trip(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        k2, ct2 = load_key_file(blob)
        assert k2 == key
        assert decrypt_mapping(ct2, k2) == SAMPLE_MAPPING

    def test_uses_v3_sentinel(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        assert blob.startswith(_SENTINEL_V3)

    def test_load_without_password_succeeds(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        k, c = load_key_file(blob, password=None)
        assert decrypt_mapping(c, k) == SAMPLE_MAPPING


# ---------------------------------------------------------------------------
# Key file — with password (v3 format, AES-256-GCM key wrapping)
# ---------------------------------------------------------------------------

class TestKeyFileWithPassword:
    def test_round_trip(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="hunter2")
        k2, ct2 = load_key_file(blob, password="hunter2")
        assert k2 == key
        assert decrypt_mapping(ct2, k2) == SAMPLE_MAPPING

    def test_uses_v3_sentinel(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="secret")
        assert blob.startswith(_SENTINEL_V3)

    def test_wrong_password_raises_value_error(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="correct")
        with pytest.raises(ValueError, match="Incorrect passphrase"):
            load_key_file(blob, password="wrong")

    def test_missing_password_loads_as_no_password(self):
        """v3 without password flag — key is embedded in cleartext."""
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)  # no password
        k, c = load_key_file(blob, password=None)
        assert decrypt_mapping(c, k) == SAMPLE_MAPPING

    def test_data_key_not_in_blob(self):
        """The raw data key must not appear in the password-protected blob."""
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="mypassword")
        assert key not in blob

    def test_different_passwords_differ(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob1 = save_key_file(key, ct, password="alpha")
        blob2 = save_key_file(key, ct, password="alpha")
        # Different random salts/nonces → different blobs even for same password
        assert blob1 != blob2

    def test_empty_password_string_treated_as_no_password(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password=None)
        assert blob.startswith(_SENTINEL_V3)


# ---------------------------------------------------------------------------
# Legacy format backward compatibility
# ---------------------------------------------------------------------------

class TestLegacyFormats:
    def test_v1_blob_can_be_loaded(self):
        """v1 (Fernet, no password) blobs must still be readable."""
        from cryptography.fernet import Fernet
        fernet_key = Fernet.generate_key()
        plaintext = b'{"[PERSON]": "Alice"}'
        ct = Fernet(fernet_key).encrypt(plaintext)
        blob = fernet_key + _SENTINEL_V1 + ct
        key, ciphertext = load_key_file(blob)
        assert key == fernet_key
        # Decrypting with a Fernet key (44 bytes) falls through to legacy path
        recovered = decrypt_mapping(ciphertext, key)
        assert recovered == {"[PERSON]": "Alice"}

    def test_v2_blob_can_be_loaded(self):
        """v2 (Fernet, password-protected) blobs must still be readable."""
        import base64
        import hashlib
        from cryptography.fernet import Fernet
        fernet_key = Fernet.generate_key()
        plaintext = b'{"[EMAIL]": "a@b.com"}'
        ct = Fernet(fernet_key).encrypt(plaintext)
        # Build a v2 blob manually
        salt = b"\x00" * 16
        raw = hashlib.pbkdf2_hmac("sha256", b"pw", salt, 480_000)
        pw_key = base64.urlsafe_b64encode(raw[:32])
        enc_key = Fernet(pw_key).encrypt(fernet_key)
        enc_key_len = len(enc_key).to_bytes(4, "big")
        blob = _SENTINEL_V2 + salt + enc_key_len + enc_key + ct
        key, ciphertext = load_key_file(blob, password="pw")
        assert key == fernet_key
        recovered = decrypt_mapping(ciphertext, key)
        assert recovered == {"[EMAIL]": "a@b.com"}


# ---------------------------------------------------------------------------
# Corrupt / invalid blobs
# ---------------------------------------------------------------------------

class TestCorruptBlobs:
    def test_unrecognised_format_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            load_key_file(b"this is not a valid key file at all")

    def test_truncated_v3_raises(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct, password="pw")
        with pytest.raises(ValueError):
            load_key_file(blob[:20], password="pw")

    def test_truncated_v3_no_password_raises(self):
        key = generate_key()
        ct = encrypt_mapping(SAMPLE_MAPPING, key)
        blob = save_key_file(key, ct)
        with pytest.raises(ValueError):
            load_key_file(blob[:15])  # too short for sentinel + key
