"""Privacy risk summariser — generates a one-sentence human-readable summary.

Uses MBZUAI/LaMini-Flan-T5-77M locally for natural-language generation,
with a deterministic template fallback if the model is unavailable.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from gocalma.regex_patterns import CRITICAL_TYPES

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_MODEL_ID = "MBZUAI/LaMini-Flan-T5-77M"
_pipe = None
_pipe_lock = threading.Lock()


def _get_pipe():
    """Lazily load and cache the text2text-generation pipeline."""
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                from transformers import pipeline
                _pipe = pipeline(
                    "text2text-generation",
                    model=_MODEL_ID,
                    max_length=80,
                )
    return _pipe


def is_model_cached() -> bool:
    """Return True if LaMini-Flan-T5-77M weights exist in HuggingFace cache."""
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        return any(cache_dir.glob("models--MBZUAI--LaMini-Flan-T5-77M"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Severity logic (deterministic — no model needed)
# ---------------------------------------------------------------------------

def _compute_severity(counts: dict[str, int]) -> str:
    """Return 'critical', 'moderate', or 'low' based on entity types found."""
    if any(t in CRITICAL_TYPES for t in counts):
        return "critical"
    person_loc = counts.get("PERSON", 0) + counts.get("LOCATION", 0)
    if person_loc > 2:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Fallback template (used when model unavailable)
# ---------------------------------------------------------------------------

_TYPE_LABELS: dict[str, str] = {
    "CH_AHV": "AHV number",
    "US_SSN": "SSN",
    "IBAN_CH": "Swiss IBAN",
    "IBAN_INTL": "IBAN",
    "CREDIT_CARD": "credit card number",
    "PERSON": "name",
    "LOCATION": "location",
    "ADDRESS": "address",
    "EMAIL": "email address",
    "PHONE_INTL": "phone number",
    "PHONE_NUMBER": "phone number",
    "DATE_OF_BIRTH": "date of birth",
    "DATE_TIME": "date",
    "INSURANCE_NUMBER": "insurance number",
    "CH_ZUGANGSCODE": "access code",
    "CH_ID_NUMBER": "ID number",
    "CH_POSTAL": "postal code",
    "ORGANIZATION": "organisation",
    "IP_ADDRESS": "IP address",
}


def _pluralise(label: str, count: int) -> str:
    if count == 1:
        return f"1 {label}"
    # Simple English pluralisation
    if label.endswith("s") or label.endswith("x") or label.endswith("address"):
        return f"{count} {label}es"
    return f"{count} {label}s"


def _template_sentence(counts: dict[str, int]) -> str:
    """Build a fallback summary sentence from counts."""
    if not counts:
        return "No personally identifiable information was detected."
    parts = []
    for entity_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        label = _TYPE_LABELS.get(entity_type, entity_type.lower().replace("_", " "))
        parts.append(_pluralise(label, count))
    joined = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    return f"This document contains {joined}."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_summary(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a privacy risk summary from detected entities.

    Parameters
    ----------
    entities : list[dict]
        Each dict must have a ``"type"`` key (entity type string).

    Returns
    -------
    dict with keys: sentence, severity, critical_types_found, counts
    """
    # Count entity types
    counts: dict[str, int] = dict(Counter(e.get("type", "UNKNOWN") for e in entities))

    severity = _compute_severity(counts)
    critical_found = sorted(t for t in counts if t in CRITICAL_TYPES)

    # Build a readable input for the model
    fallback = _template_sentence(counts)

    sentence = None
    try:
        pipe = _get_pipe()
        prompt = (
            f"Rewrite this sentence to sound natural: \"{fallback}\" "
            f"Keep it to one sentence."
        )
        result = pipe(prompt)
        generated = result[0].get("generated_text", "").strip()
        # Accept only if it looks like a real sentence (not a refusal or empty)
        if generated and len(generated) > 10 and "sorry" not in generated.lower():
            sentence = generated
    except Exception as exc:
        _log.warning("Summariser model failed, using template fallback: %s", exc)

    if not sentence:
        sentence = fallback

    return {
        "sentence": sentence,
        "severity": severity,
        "critical_types_found": critical_found,
        "counts": counts,
    }
