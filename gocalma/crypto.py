"""Encrypted key mapping for reversible redactions.

Stores a JSON mapping of {redaction_label -> original_text}, encrypted with
AES-256-GCM (authenticated encryption).

Security model
--------------
Without a password  (legacy / convenience mode):
    A random 256-bit key is generated and bundled with the ciphertext in the
    .gocalma file.  The file itself must be kept private; anyone who has it
    can decrypt the mapping.  Suitable for personal use on a single machine.

With a password (recommended):
    The data key is encrypted with a key derived from the user's passphrase
    via PBKDF2-HMAC-SHA256 (480 000 iterations, 256-bit output).  The
    .gocalma file then contains only [salt + encrypted_key + ciphertext] —
    without the passphrase the mapping cannot be recovered even if the file
    leaks.

Current format — v3 (AES-256-GCM, sentinel b"::GOCALMA3::"):

  Password-protected:
    SENTINEL   (12 bytes)
    salt       (16 bytes)
    enc_key_nonce (12 bytes)
    enc_key_len   (4 bytes, big-endian)
    enc_key    (variable — AES-256-GCM encrypted data key)
    data_nonce (12 bytes)
    ciphertext (remainder — AES-256-GCM encrypted JSON mapping)

  No password:
    SENTINEL   (12 bytes)
    raw_key    (32 bytes)
    data_nonce (12 bytes)
    ciphertext (remainder)

Legacy formats (v1, v2 — Fernet / AES-128-CBC) are still readable for
backward compatibility but new files always use v3.
"""

from __future__ import annotations

import hashlib
import json
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Fernet is only imported when reading legacy v1/v2 files
_fernet_mod = None

def _get_fernet():
    global _fernet_mod
    if _fernet_mod is None:
        from cryptography import fernet as _fm
        _fernet_mod = _fm
    return _fernet_mod

_SENTINEL_V1 = b"::GOCALMA::"   # legacy / no-password (Fernet)
_SENTINEL_V2 = b"::GOCALMA2::"  # legacy / password-protected (Fernet)
_SENTINEL_V3 = b"::GOCALMA3::"  # current — AES-256-GCM
_SALT_SIZE = 16
_NONCE_SIZE = 12   # 96-bit nonce for GCM
_KEY_SIZE = 32     # 256-bit key
_PBKDF2_ITERATIONS = 480_000


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte (256-bit) key from *password* using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
        dklen=_KEY_SIZE,
    )


def _derive_fernet_key(password: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible key for reading legacy v2 files."""
    import base64
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
    """Generate a new random 256-bit data key (32 bytes)."""
    return os.urandom(_KEY_SIZE)


def encrypt_mapping(mapping: dict[str, str], key: bytes) -> bytes:
    """Encrypt *mapping* with *key* using AES-256-GCM.

    Returns nonce (12 bytes) + ciphertext+tag.
    """
    plaintext = json.dumps(mapping).encode("utf-8")
    nonce = os.urandom(_NONCE_SIZE)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct


def decrypt_mapping(ciphertext: bytes, key: bytes) -> dict[str, str]:
    """Decrypt *ciphertext* with *key* and return the mapping dict.

    Accepts both v3 (AES-256-GCM: nonce + ct) and legacy Fernet format.
    Raises ``ValueError`` if the key is wrong or the data is corrupt.
    """
    # v3 format: first 12 bytes are nonce
    if len(key) == _KEY_SIZE:
        if len(ciphertext) < _NONCE_SIZE + 16:
            raise ValueError("Ciphertext is too short.")
        nonce = ciphertext[:_NONCE_SIZE]
        ct = ciphertext[_NONCE_SIZE:]
        try:
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ct, None)
            return json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Decryption failed: {exc}") from exc

    # Legacy Fernet key (44 bytes, base64)
    fernet = _get_fernet()
    try:
        plaintext = fernet.Fernet(key).decrypt(ciphertext)
        return json.loads(plaintext.decode("utf-8"))
    except fernet.InvalidToken as exc:
        raise ValueError(f"Decryption failed: {exc}") from exc


def save_key_file(
    key: bytes,
    ciphertext: bytes,
    password: str | None = None,
) -> bytes:
    """Bundle key + ciphertext into a single downloadable blob.

    If *password* is provided the data key is itself encrypted with a
    PBKDF2-derived key so the file is useless without the passphrase.
    Always uses v3 (AES-256-GCM) format.
    """
    if password:
        salt = os.urandom(_SALT_SIZE)
        pw_key = _derive_key(password, salt)
        nonce_key = os.urandom(_NONCE_SIZE)
        aesgcm = AESGCM(pw_key)
        enc_key = aesgcm.encrypt(nonce_key, key, None)
        enc_key_len = len(enc_key).to_bytes(4, "big")
        return (
            _SENTINEL_V3 + salt + nonce_key
            + enc_key_len + enc_key + ciphertext
        )

    # No password: sentinel + raw key + ciphertext
    return _SENTINEL_V3 + key + ciphertext


def load_key_file(
    blob: bytes,
    password: str | None = None,
) -> tuple[bytes, bytes]:
    """Parse a key-file blob and return ``(key, ciphertext)``.

    Supports v3 (AES-256-GCM), v2 (Fernet + password), and v1 (Fernet,
    no password) formats.  Raises ``ValueError`` for format errors or
    wrong passwords.
    """
    # --- v3: AES-256-GCM ---
    if blob.startswith(_SENTINEL_V3):
        rest = blob[len(_SENTINEL_V3):]
        if password:
            # Password-protected v3
            min_len = _SALT_SIZE + _NONCE_SIZE + 4
            if len(rest) < min_len:
                raise ValueError("Key file is truncated or corrupt.")
            salt = rest[:_SALT_SIZE]
            rest = rest[_SALT_SIZE:]
            nonce_key = rest[:_NONCE_SIZE]
            rest = rest[_NONCE_SIZE:]
            enc_key_len = int.from_bytes(rest[:4], "big")
            rest = rest[4:]
            if len(rest) < enc_key_len:
                raise ValueError("Key file is truncated or corrupt.")
            enc_key = rest[:enc_key_len]
            ciphertext = rest[enc_key_len:]
            pw_key = _derive_key(password, salt)
            try:
                aesgcm = AESGCM(pw_key)
                key = aesgcm.decrypt(nonce_key, enc_key, None)
            except Exception:
                raise ValueError("Incorrect passphrase or corrupted key file.")
            return key, ciphertext
        else:
            # No password v3: raw key follows sentinel
            if len(rest) < _KEY_SIZE:
                raise ValueError("Key file is truncated or corrupt.")
            key = rest[:_KEY_SIZE]
            ciphertext = rest[_KEY_SIZE:]
            return key, ciphertext

    # --- v2: Fernet + password (legacy) ---
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
        fernet = _get_fernet()
        pw_key = _derive_fernet_key(password, salt)
        try:
            key = fernet.Fernet(pw_key).decrypt(enc_key)
        except fernet.InvalidToken:
            raise ValueError("Incorrect passphrase or corrupted key file.")
        return key, ciphertext

    # --- v1: Fernet, no password (legacy) ---
    if _SENTINEL_V1 not in blob:
        raise ValueError("Unrecognised key file format.")
    idx = blob.index(_SENTINEL_V1)
    key = blob[:idx]
    ciphertext = blob[idx + len(_SENTINEL_V1):]
    return key, ciphertext
