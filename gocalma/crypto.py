"""Encrypted key mapping for reversible redactions.

Stores a JSON mapping of {redaction_label -> original_text}, encrypted with
Fernet (AES-128-CBC + HMAC-SHA256).

Security model
--------------
Without a password  (legacy / convenience mode):
    A random Fernet key is generated and bundled with the ciphertext in the
    .gocalma file.  The file itself must be kept private; anyone who has it
    can decrypt the mapping.  Suitable for personal use on a single machine.

With a password (recommended):
    The Fernet key is itself encrypted with a key derived from the user's
    passphrase via PBKDF2-HMAC-SHA256 (480 000 iterations).  The .gocalma
    file then contains only [salt + encrypted_key + ciphertext] — without
    the passphrase the mapping cannot be recovered even if the file leaks.

File format (password-protected, sentinel b"::GOCALMA2::"):
    SENTINEL (12 bytes)
    salt     (16 bytes)
    enc_key_len (4 bytes, big-endian)
    enc_key  (variable — Fernet token wrapping the Fernet data key)
    ciphertext (remainder)

File format (no password, legacy sentinel b"::GOCALMA::"):
    key (44 bytes, URL-safe base64 Fernet key)
    SENTINEL (11 bytes)
    ciphertext (remainder)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os

from cryptography.fernet import Fernet, InvalidToken

_SENTINEL_V1 = b"::GOCALMA::"   # legacy / no-password format
_SENTINEL_V2 = b"::GOCALMA2::"  # password-protected format
_SALT_SIZE = 16
_PBKDF2_ITERATIONS = 480_000


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte Fernet-compatible key from *password* using PBKDF2."""
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(raw[:32])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_key() -> bytes:
    """Generate a new random Fernet data key."""
    return Fernet.generate_key()


def encrypt_mapping(mapping: dict[str, str], key: bytes) -> bytes:
    """Encrypt *mapping* with *key* and return ciphertext bytes."""
    plaintext = json.dumps(mapping).encode("utf-8")
    return Fernet(key).encrypt(plaintext)


def decrypt_mapping(ciphertext: bytes, key: bytes) -> dict[str, str]:
    """Decrypt *ciphertext* with *key* and return the mapping dict.

    Raises ``cryptography.fernet.InvalidToken`` if the key is wrong or the
    data is corrupt.
    """
    plaintext = Fernet(key).decrypt(ciphertext)
    return json.loads(plaintext.decode("utf-8"))


def save_key_file(
    key: bytes,
    ciphertext: bytes,
    password: str | None = None,
) -> bytes:
    """Bundle key + ciphertext into a single downloadable blob.

    If *password* is provided the data key is itself encrypted with a
    PBKDF2-derived key so the file is useless without the passphrase.
    """
    if password:
        salt = os.urandom(_SALT_SIZE)
        pw_key = _derive_key(password, salt)
        enc_key = Fernet(pw_key).encrypt(key)
        enc_key_len = len(enc_key).to_bytes(4, "big")
        return _SENTINEL_V2 + salt + enc_key_len + enc_key + ciphertext

    # Legacy / no-password: key + sentinel + ciphertext
    return key + _SENTINEL_V1 + ciphertext


def load_key_file(
    blob: bytes,
    password: str | None = None,
) -> tuple[bytes, bytes]:
    """Parse a key-file blob and return ``(key, ciphertext)``.

    Raises ``ValueError`` for format errors or wrong passwords.
    """
    if blob.startswith(_SENTINEL_V2):
        rest = blob[len(_SENTINEL_V2):]
        if len(rest) < _SALT_SIZE + 4:
            raise ValueError("Key file is truncated or corrupt.")
        salt = rest[:_SALT_SIZE]
        rest = rest[_SALT_SIZE:]
        enc_key_len = int.from_bytes(rest[:4], "big")
        rest = rest[4:]
        if len(rest) < enc_key_len:
            raise ValueError("Key file is truncated or corrupt.")
        enc_key = rest[:enc_key_len]
        ciphertext = rest[enc_key_len:]
        if password is None:
            raise ValueError(
                "This key file is password-protected. "
                "Enter the passphrase used when the file was created."
            )
        pw_key = _derive_key(password, salt)
        try:
            key = Fernet(pw_key).decrypt(enc_key)
        except InvalidToken:
            raise ValueError("Incorrect passphrase or corrupted key file.")
        return key, ciphertext

    # Legacy format
    if _SENTINEL_V1 not in blob:
        raise ValueError("Unrecognised key file format.")
    idx = blob.index(_SENTINEL_V1)
    key = blob[:idx]
    ciphertext = blob[idx + len(_SENTINEL_V1):]
    return key, ciphertext
