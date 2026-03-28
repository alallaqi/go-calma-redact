"""GDPR-style audit trail for redaction operations.

Writes metadata only — no document content is ever stored.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from gocalma.regex_patterns import CRITICAL_TYPES

APP_VERSION = "1.0.0"

# Severity per entity type (mirrors regex_patterns priority scale)
_SEVERITY: dict[str, str] = {t: "critical" for t in CRITICAL_TYPES}
_MODERATE_TYPES = {"PERSON", "LOCATION", "ADDRESS", "EMAIL", "PHONE_INTL", "PHONE_NUMBER"}


def _entity_severity(entity_type: str) -> str:
    if entity_type in _SEVERITY:
        return "critical"
    if entity_type in _MODERATE_TYPES:
        return "moderate"
    return "low"


def create_audit(
    entities: list[Any],
    action_taken: str,
    model_used: str,
    llm_used: bool,
    filename: str,
) -> dict[str, Any]:
    """Build an audit record dict from redaction metadata.

    Parameters
    ----------
    entities : list
        Detected entities — each must have an ``entity_type`` attribute or
        a ``"type"`` key.
    action_taken : str
        The redaction action: redact, replace, mask, hash, encrypt, highlight.
    model_used : str
        NER model identifier.
    llm_used : bool
        Whether LLM verification ran successfully.
    filename : str
        Original filename (hashed — never stored in plain text).
    """
    # Count by type
    counts: dict[str, int] = {}
    for ent in entities:
        etype = getattr(ent, "entity_type", None) or ent.get("type", "UNKNOWN")
        counts[etype] = counts.get(etype, 0) + 1

    entity_summary = {
        etype: {"count": count, "severity": _entity_severity(etype)}
        for etype, count in sorted(counts.items())
    }

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filename_hash": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
        "model": model_used,
        "llm_verified": llm_used,
        "entity_summary": entity_summary,
        "action": action_taken,
        "total_entities_redacted": sum(counts.values()),
        "app_version": APP_VERSION,
    }


def save_audit(audit_record: dict[str, Any], output_path: str) -> str:
    """Write *audit_record* as JSON to *output_path*. Returns the path."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2, ensure_ascii=False)
    return output_path


def audit_to_bytes(audit_record: dict[str, Any]) -> bytes:
    """Serialise *audit_record* to UTF-8 JSON bytes (for in-memory download)."""
    return json.dumps(audit_record, indent=2, ensure_ascii=False).encode("utf-8")
