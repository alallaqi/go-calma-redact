"""Centralised PII regex patterns – Swiss, international, and universal.

Every pattern is a (compiled_regex, entity_type, priority) tuple.
Priority scale:
    10 = highly specific identifiers (AHV, IBAN, SSN, credit card)
     9 = email, passport, insurance, Zugangscode
     8 = phone numbers
     7 = label-context (keyword + value pairs)
     6 = NER person name (used later for merge)
     5 = NER location / address
     4 = NER organisation
     3 = date of birth
     2 = health data (ICD codes, diagnoses)
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Critical entity types — never downgrade or suppress these
# ---------------------------------------------------------------------------
CRITICAL_TYPES: set[str] = {
    "CH_AHV",
    "US_SSN",
    "IBAN_CH",
    "IBAN_INTL",
    "CREDIT_CARD",
}

# ---------------------------------------------------------------------------
# Luhn checksum for credit card validation
# ---------------------------------------------------------------------------
def _luhn_check(digits: str) -> bool:
    """Return True if *digits* passes the Luhn checksum (credit card validation)."""
    nums = [int(d) for d in digits if d.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Pattern registry: (compiled_regex, entity_type, priority)
# ---------------------------------------------------------------------------
PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    # ── Swiss ──────────────────────────────────────────────────────────────
    # AHV / OASI number  756.XXXX.XXXX.XX
    (re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b"), "CH_AHV", 10),
    (re.compile(r"\b756[\s,]\d{4}[\s,]\d{4}[\s,]\d{2}\b"), "CH_AHV", 10),

    # Swiss IBAN  CH56 0483 5012 3456 7800 9
    (re.compile(r"\bCH\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{1}\b"), "IBAN_CH", 10),

    # Zugangscode  ABCD-EFgh-IJKL-MNop
    (re.compile(r"\b[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}\b"), "CH_ZUGANGSCODE", 9),

    # Swiss postal code  4-digit + city
    (re.compile(r"\b[1-9]\d{3}(?=\s+[A-ZÄÖÜ][a-zäöüß])"), "CH_POSTAL", 5),

    # Swiss personal / document ID  XX-XXXX-XX
    (re.compile(r"\b\d{2}-\d{4}-\d{2}\b"), "CH_ID_NUMBER", 9),

    # Swiss reference ID (10–13 digit number, min 10 digits = safe from short codes)
    (re.compile(r"\b\d{10,13}\b"), "CH_REFERENCE_ID", 9),

    # Insurance / policy number — explicit 3-3-3 grouped format (e.g. "100 452 956")
    # Only this high-confidence format; no bare 9-digit match to avoid false positives
    (re.compile(r"\b\d{3}\s\d{3}\s\d{3}\b"), "INSURANCE_NUMBER", 9),

    # ── International ─────────────────────────────────────────────────────
    # US Social Security Number  123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "US_SSN", 10),

    # UK National Insurance  AB123456C
    (re.compile(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\d{6}[A-D]\b"), "UK_NI", 9),

    # UK Postcode  SW1A 1AA
    (re.compile(
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
        re.IGNORECASE,
    ), "UK_POSTCODE", 5),

    # German Steuer-ID  11 digits
    (re.compile(r"\b\d{11}\b"), "DE_STEUER_ID", 9),

    # French NIR (social security)  15 digits starting with 1 or 2
    (re.compile(r"\b[12]\d{14}\b"), "FR_NIR", 9),

    # Italian Codice Fiscale  RSSMRA85M01H501Z
    (re.compile(
        r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
        re.IGNORECASE,
    ), "IT_CODICE_FISCALE", 9),

    # Spanish DNI  12345678Z
    (re.compile(r"\b\d{8}[A-Z]\b"), "ES_DNI", 9),

    # Spanish NIE  X1234567L
    (re.compile(r"\b[XYZ]\d{7}[A-Z]\b", re.IGNORECASE), "ES_NIE", 9),

    # ICAO machine-readable passport number  2 letters + 6-9 digits
    (re.compile(r"\b[A-Z]{2}\d{6,9}\b"), "ICAO_PASSPORT", 9),

    # ── Universal ─────────────────────────────────────────────────────────
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"), "EMAIL", 9),

    # International phone  E.164: +41 79 123 45 67 or +14155552671
    (re.compile(r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{0,4}\b"), "PHONE_INTL", 8),

    # International IBAN  2-letter country + 2 check digits + up to 30 alphanumeric
    (re.compile(r"\b[A-Z]{2}\d{2}\s?[\dA-Z]{4}(?:\s?[\dA-Z]{4}){2,7}(?:\s?[\dA-Z]{1,4})?\b"), "IBAN_INTL", 10),

    # Credit card  13-19 digits optionally separated by spaces/dashes (Luhn-validated)
    (re.compile(r"\b(?:\d[\s\-]?){13,19}\b"), "CREDIT_CARD", 10),

    # IP address  IPv4
    (re.compile(
        r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ), "IP_ADDRESS", 8),

    # Date of birth  DD.MM.YYYY and YYYY-MM-DD
    (re.compile(r"\b(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.\d{4}\b"), "DATE_OF_BIRTH", 3),
    (re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"), "DATE_OF_BIRTH", 3),
]

# ---------------------------------------------------------------------------
# Label-context patterns — keyword + value pairs (priority 7)
# These detect PII via surrounding labels like "Name:", "Patient:", etc.
# ---------------------------------------------------------------------------
LABEL_CONTEXT_PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    # Person name after label keywords (multilingual)
    (re.compile(
        r"(?:(?:name|patient|holder|insured|versicherte[rn]?|assur[eé]e?"
        r"|empfänger|inhaber|beneficiary|claimant)"
        r"\s*[:=\-]\s*)"
        r"([A-ZÄÖÜ][a-zäöüß]{2,25}(?:\s+[A-ZÄÖÜ][a-zäöüß]{2,25}){0,2})",
        re.IGNORECASE,
    ), "PERSON", 7),

    # Doctor / physician name after label
    (re.compile(
        r"(?:(?:physician|doctor|arzt|ärztin|médecin|dottore)"
        r"\s*[:=\-]\s*)"
        r"([A-ZÄÖÜ][a-zäöüß]{2,25}(?:\s+[A-ZÄÖÜ][a-zäöüß]{2,25}){0,2})",
        re.IGNORECASE,
    ), "PERSON", 7),

    # Insurance / policy number after label
    (re.compile(
        r"(?:(?:insurance|versicherungs?|police|policy|assurance)"
        r"[\s\-]*(?:no\.?|nr\.?|number|nummer)?"
        r"\s*[:=\-]\s*)"
        r"([A-Z0-9][\w.\-\s]{2,20})",
        re.IGNORECASE,
    ), "INSURANCE_NUMBER", 7),

    # Date of birth after label
    (re.compile(
        r"(?:(?:date\s+of\s+birth|d\.?\s*o\.?\s*b\.?|geburtsdatum"
        r"|date\s+de\s+naissance|data\s+di\s+nascita|geboren)"
        r"\s*[:=\-]\s*)"
        r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})",
        re.IGNORECASE,
    ), "DATE_OF_BIRTH", 7),

    # ID / passport number after label
    (re.compile(
        r"(?:(?:passport|ausweis|reisepass|carta\s+d'identit[àa]"
        r"|identity\s+card|personalausweis)"
        r"[\s\-]*(?:no\.?|nr\.?|number|nummer)?"
        r"\s*[:=\-]\s*)"
        r"([A-Z0-9][\w\-]{4,15})",
        re.IGNORECASE,
    ), "ID_NUMBER", 7),

    # Address after label
    (re.compile(
        r"(?:(?:address|adresse|anschrift|wohnort|domicile|indirizzo)"
        r"\s*[:=\-]\s*)"
        r"(.{5,80}?)(?=\n|\r|$)",
        re.IGNORECASE,
    ), "ADDRESS", 7),
]

# ---------------------------------------------------------------------------
# Health / medical patterns (priority 2–7)
# ---------------------------------------------------------------------------
HEALTH_PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    # ICD-10 codes  A00-T98, V-Z codes
    (re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b"), "ICD_CODE", 2),

    # Diagnosis after label
    (re.compile(
        r"(?:(?:diagnosis|diagnose|diagnostic)"
        r"\s*[:=\-]\s*)"
        r"(.{3,60}?)(?=\n|\r|$)",
        re.IGNORECASE,
    ), "HEALTH_DATA", 7),

    # Medication after label
    (re.compile(
        r"(?:(?:medications?|medikamente?|médicaments?|current\s+treatment)"
        r"\s*[:=\-]\s*)"
        r"(.{3,80}?)(?=\n|\r|$)",
        re.IGNORECASE,
    ), "HEALTH_DATA", 7),

    # Allergy after label
    (re.compile(
        r"(?:(?:allergies?|allergi(?:en|es?))"
        r"\s*[:=\-]\s*)"
        r"(.{3,60}?)(?=\n|\r|$)",
        re.IGNORECASE,
    ), "HEALTH_DATA", 7),

    # Blood type
    (re.compile(
        r"(?:(?:blood\s+(?:type|group)|blutgruppe|groupe\s+sanguin)"
        r"\s*[:=\-]?\s*)"
        r"([ABO]{1,2}[+\-])",
        re.IGNORECASE,
    ), "HEALTH_DATA", 7),
]


# ---------------------------------------------------------------------------
# run_regex — scan text with all patterns
# ---------------------------------------------------------------------------
def run_regex(text: str) -> list[dict[str, Any]]:
    """Run every pattern against *text* and return a flat list of matches.

    Each match is a dict with keys:
        text, start, end, type, priority, source
    """
    results: list[dict[str, Any]] = []

    # Standard patterns (full match)
    for regex, entity_type, priority in PATTERNS:
        for m in regex.finditer(text):
            matched_text = m.group()
            # Luhn validation for credit card matches
            if entity_type == "CREDIT_CARD":
                digits = "".join(c for c in matched_text if c.isdigit())
                if not _luhn_check(digits):
                    continue
            results.append({
                "text": matched_text,
                "start": m.start(),
                "end": m.end(),
                "type": entity_type,
                "priority": priority,
                "source": "regex",
            })

    # Label-context patterns (capture group 1)
    for regex, entity_type, priority in LABEL_CONTEXT_PATTERNS:
        for m in regex.finditer(text):
            captured = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group().strip()
            if len(captured) < 3:
                continue
            results.append({
                "text": captured,
                "start": m.start(1) if m.lastindex and m.lastindex >= 1 else m.start(),
                "end": m.end(1) if m.lastindex and m.lastindex >= 1 else m.end(),
                "type": entity_type,
                "priority": priority,
                "source": "regex",
            })

    # Health patterns (capture group 1 where applicable)
    for regex, entity_type, priority in HEALTH_PATTERNS:
        for m in regex.finditer(text):
            if m.lastindex and m.lastindex >= 1:
                captured = m.group(1).strip()
                start = m.start(1)
                end = m.end(1)
            else:
                captured = m.group().strip()
                start = m.start()
                end = m.end()
            if len(captured) < 2:
                continue
            results.append({
                "text": captured,
                "start": start,
                "end": end,
                "type": entity_type,
                "priority": priority,
                "source": "regex",
            })

    return results


# ---------------------------------------------------------------------------
# merge_with_priority — combine regex + NER, resolve overlaps
# ---------------------------------------------------------------------------
def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return True if the two spans share at least one character."""
    return a_start < b_end and b_start < a_end


def merge_with_priority(
    regex_entities: list[dict[str, Any]],
    ner_entities: list[dict[str, Any]],
    llm_entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge regex, NER, and LLM entity lists, resolving duplicates.

    Deduplication is two-pass:
    1. **Span overlap** — if two entities share any character positions,
       keep only the highest-priority one (regex wins ties).
    2. **Text identity** — after span dedup, if two surviving entities have
       the same text (case-insensitive, stripped), keep only the first
       (highest-priority) one.  This catches the LLM re-adding something
       regex or NER already found at a different span offset.
    """
    # Tag NER entities with defaults
    for ent in ner_entities:
        ent.setdefault("source", "ner")
        ent.setdefault("priority", _default_ner_priority(ent.get("type", "")))

    # Tag LLM entities with defaults
    for ent in (llm_entities or []):
        ent.setdefault("source", "llm")
        ent.setdefault("priority", _default_ner_priority(ent.get("type", "")))

    all_entities = list(regex_entities) + list(ner_entities) + list(llm_entities or [])

    # Sort: highest priority first, then regex > ner > llm, then earliest span
    _SOURCE_RANK = {"regex": 0, "ner": 1, "llm": 2}
    all_entities.sort(key=lambda e: (
        -e.get("priority", 0),
        _SOURCE_RANK.get(e.get("source", "ner"), 1),
        e.get("start", 0),
    ))

    # Pass 1: span overlap dedup
    kept: list[dict[str, Any]] = []
    for candidate in all_entities:
        c_start = candidate["start"]
        c_end = candidate["end"]
        overlaps_span = any(
            _spans_overlap(c_start, c_end, accepted["start"], accepted["end"])
            for accepted in kept
        )
        if not overlaps_span:
            kept.append(candidate)

    # Pass 2: text identity dedup (case-insensitive, stripped)
    seen_texts: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for ent in kept:
        normalised = ent["text"].strip().lower()
        if normalised not in seen_texts:
            seen_texts.add(normalised)
            deduped.append(ent)

    deduped.sort(key=lambda e: e["start"])
    return deduped


def filter_implausible(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove entities that are too short, purely punctuation, or trivial.

    Discards any entity where the stripped text is:
    - Less than 3 characters long
    - Only whitespace
    - Only digits and under 3 chars
    - Only punctuation
    """
    result: list[dict[str, Any]] = []
    for ent in entities:
        text = ent.get("text", "").strip()
        if len(text) < 3:
            continue
        if not any(c.isalnum() for c in text):
            continue  # purely punctuation / symbols
        result.append(ent)
    return result


def _default_ner_priority(entity_type: str) -> int:
    """Assign a default priority to NER entities based on type."""
    _NER_PRIORITIES: dict[str, int] = {
        "PERSON": 6,
        "LOCATION": 5,
        "ADDRESS": 5,
        "ORGANIZATION": 4,
        "DATE_TIME": 3,
        "DATE_OF_BIRTH": 3,
        "PHONE_NUMBER": 8,
        "EMAIL_ADDRESS": 9,
        "IBAN_CODE": 10,
        "CREDIT_CARD": 10,
        "US_SSN": 10,
        "CH_AHV": 10,
        "HEALTH_DATA": 2,
        "ICD_CODE": 2,
    }
    return _NER_PRIORITIES.get(entity_type, 5)
