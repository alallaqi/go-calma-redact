"""PII detection — multilingual BERT NER + centralised regex patterns.

Uses a single multilingual model (Davlan/bert-base-multilingual-cased-ner-hrl)
loaded once as a lazy singleton.  All regex patterns live in regex_patterns.py.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import re as _re

from gocalma.regex_patterns import run_regex, merge_with_priority, filter_implausible

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single NER model — loaded once on first call
# ---------------------------------------------------------------------------

_NER_LABEL_MAP: dict[str, str] = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE_TIME",
}

_ner_pipe = None
_ner_pipe_model: str | None = None
_ner_pipe_lock = threading.Lock()

# Currently selected NER model — changed via set_ner_model()
_active_ner_model: str = "Davlan/bert-base-multilingual-cased-ner-hrl"


def _get_ner_pipe():
    """Lazily load the NER pipeline for the active model (thread-safe singleton)."""
    global _ner_pipe, _ner_pipe_model
    if _ner_pipe is None or _ner_pipe_model != _active_ner_model:
        with _ner_pipe_lock:
            if _ner_pipe is None or _ner_pipe_model != _active_ner_model:
                from transformers import pipeline
                _ner_pipe = pipeline(
                    "ner",
                    model=_active_ner_model,
                    aggregation_strategy="simple",
                )
                _ner_pipe_model = _active_ner_model
    return _ner_pipe


def set_ner_model(model_id: str) -> None:
    """Set the active NER model. Takes effect on the next _get_ner_pipe() call."""
    global _active_ner_model
    if model_id in NLP_MODELS:
        _active_ner_model = model_id


def get_ner_model() -> str:
    """Return the currently active NER model ID."""
    return _active_ner_model


# Backward-compat exports — other modules import these names.
DEFAULT_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
NLP_MODELS: dict[str, dict] = {
    "Davlan/bert-base-multilingual-cased-ner-hrl": {
        "engine_name": "transformers",
        "model_name": "Davlan/bert-base-multilingual-cased-ner-hrl",
        "lang_codes": ["de", "fr", "it", "en", "es", "pt", "nl"],
        "description": "Multilingual BERT — best all-round (7 languages)",
    },
    "dslim/bert-base-NER": {
        "engine_name": "transformers",
        "model_name": "dslim/bert-base-NER",
        "lang_codes": ["en"],
        "description": "English-only BERT NER — faster, English docs only",
    },
    "Davlan/xlm-roberta-large-ner-hrl": {
        "engine_name": "transformers",
        "model_name": "Davlan/xlm-roberta-large-ner-hrl",
        "lang_codes": ["de", "fr", "it", "en", "es", "pt", "nl"],
        "description": "XLM-RoBERTa Large — highest accuracy, slower",
    },
}


def available_models() -> dict[str, dict]:
    """Return NER models whose transformers backend is installed."""
    try:
        import transformers  # noqa: F401
        return dict(NLP_MODELS)
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# PIIEntity dataclass
# ---------------------------------------------------------------------------

@dataclass
class PIIEntity:
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    page_num: int
    source: str = "NER"
    analysis: str = ""


# ---------------------------------------------------------------------------
# Language detection (for logging / display only)
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Detect language of *text*. Returns BCP-47 code or 'en' as fallback."""
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text[:2000])
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Context-aware false positive filter
# ---------------------------------------------------------------------------

_COUNTRY_NAMES = {
    "switzerland", "schweiz", "suisse", "svizzera",
    "germany", "france", "europe", "europa",
    "international", "worldwide",
}

_PRODUCT_SCOPE_WORDS = [
    "anywhere in", "throughout", "available in", "ward", "coverage",
    "valid in", "gültig in", "valable en", "service in", "protection",
]

_ABBREVIATION_CONTEXT = [
    "abbreviation", "abkürzung", "general insurance",
    "additional insurance", "federal law", "conditions",
]

_PREMIUM_REGION_RE = _re.compile(r"^[A-Z]{1,2}\s?\d{1,2}$")
_PREMIUM_REGION_CONTEXT = ["premium region", "prämienregion", "region"]

_ABBREVIATION_RE = _re.compile(r"^[A-Z]{2,4}$")


def is_contextual_false_positive(entity: dict, full_page_text: str) -> bool:
    """Return True if the entity is a false positive based on surrounding context.

    Checks three cases:
    A) Country/region name in a product-scope description
    B) Abbreviation table entries (2-4 uppercase letters near legend keywords)
    C) Premium region codes (e.g. "BE 3" near "premium region")
    """
    text = entity.get("text", "")
    etype = entity.get("type", "")
    start = entity.get("start", 0)
    end = entity.get("end", start + len(text))

    # CASE A — Country name in product description context
    if etype == "LOCATION" and text.strip().lower() in _COUNTRY_NAMES:
        before = full_page_text[max(0, start - 60):start].lower()
        if any(w in before for w in _PRODUCT_SCOPE_WORDS):
            return True

    # CASE B — Abbreviation table entries
    if _ABBREVIATION_RE.match(text.strip()):
        window_start = max(0, start - 60)
        window_end = min(len(full_page_text), end + 60)
        window = full_page_text[window_start:window_end].lower()
        if any(w in window for w in _ABBREVIATION_CONTEXT):
            return True

    # CASE C — Premium region codes
    if _PREMIUM_REGION_RE.match(text.strip()):
        window_start = max(0, start - 40)
        window_end = min(len(full_page_text), end + 40)
        window = full_page_text[window_start:window_end].lower()
        if any(w in window for w in _PREMIUM_REGION_CONTEXT):
            return True

    return False


# ---------------------------------------------------------------------------
# Computed confidence score
# ---------------------------------------------------------------------------

_PII_CONTEXT_KEYWORDS = [
    "insurance no", "policyholder", "date of birth", "geboren",
    "versicherte", "assuré", "policy", "mr", "mrs", "ms", "herr",
    "frau", "insured", "customer", "client",
]

_TYPE_FLOORS: dict[str, float] = {
    "PERSON": 0.80,
    "DATE_OF_BIRTH": 0.85,
    "EMAIL_ADDRESS": 0.95,
    "EMAIL": 0.95,
    "PHONE_NUMBER": 0.80,
    "PHONE_INTL": 0.80,
    "LOCATION": 0.60,
    "ORGANIZATION": 0.55,
    "ORG": 0.55,
}


def compute_confidence(entity: dict, full_text: str) -> float:
    """Return a 0.0–1.0 confidence score reflecting actual PII likelihood.

    Unlike the raw NER token probability, this accounts for source,
    entity type, span length, surrounding context, and repetition.
    """
    score = entity.get("score", 0.5)
    entity_type = entity["type"]
    text = entity["text"].strip()

    # 1. SOURCE BOOST — regex is deterministic, always certain
    if entity.get("source") == "regex":
        return 1.0

    # 2. TYPE FLOOR — certain types are always high confidence if NER found them
    floor = _TYPE_FLOORS.get(entity_type, 0.50)
    score = max(score, floor)

    # 3. LENGTH BOOST — longer spans are more likely correct
    word_count = len(text.split())
    if word_count >= 3:
        score = min(score + 0.15, 1.0)
    elif word_count == 2:
        score = min(score + 0.10, 1.0)

    # 4. CONTEXT BOOST — entity appears near known PII keywords
    start = max(0, entity.get("start", 0) - 80)
    end = min(len(full_text), entity.get("end", 0) + 80)
    surrounding = full_text[start:end].lower()
    if any(kw in surrounding for kw in _PII_CONTEXT_KEYWORDS):
        score = min(score + 0.10, 1.0)

    # 5. REPETITION SIGNAL — 3+ occurrences = almost certainly the subject
    occurrences = full_text.lower().count(text.lower())
    if occurrences >= 3:
        score = min(score + 0.10, 1.0)

    return round(score, 2)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

_MAX_NER_CHARS = 4500  # BERT models cap at 512 tokens


def detect_pii(
    text: str,
    page_num: int,
    language: str | None = None,
    score_threshold: float = 0.35,
    **kwargs,
) -> list[PIIEntity]:
    """Detect PII in *text* using regex patterns + multilingual BERT NER.

    1. Run regex_patterns.run_regex(text)
    2. Run multilingual BERT NER
    3. Merge with priority (regex wins ties)
    4. Return deduplicated entity list

    The *language* parameter is for logging only — the model handles all
    languages automatically.
    """
    if language is None:
        language = _detect_language(text)
    _log.debug("detect_pii: page=%d lang=%s len=%d", page_num, language, len(text))

    # Step 1: regex patterns
    regex_hits = run_regex(text)

    # Step 2: multilingual BERT NER
    ner_hits: list[dict] = []
    try:
        pipe = _get_ner_pipe()
        items = pipe(text[:_MAX_NER_CHARS])
        for item in items:
            entity_group = item.get("entity_group", item.get("entity", ""))
            if entity_group.startswith(("B-", "I-")):
                entity_group = entity_group[2:]
            entity_type = _NER_LABEL_MAP.get(entity_group, entity_group)
            score = float(item["score"])
            if score < score_threshold:
                continue
            ner_hits.append({
                "text": text[item["start"]:item["end"]],
                "start": item["start"],
                "end": item["end"],
                "type": entity_type,
                "source": "ner",
            })
    except Exception as exc:
        _log.warning("NER pipeline failed: %s", exc)

    # Step 3: merge (regex wins ties on overlapping spans)
    merged = merge_with_priority(regex_hits, ner_hits)

    # Step 4: context-aware false positive filter
    merged = [
        hit for hit in merged
        if not is_contextual_false_positive(hit, text)
    ]

    # Step 5: remove implausibly short / trivial entities
    merged = filter_implausible(merged)

    # Step 5b: compute confidence scores
    for hit in merged:
        hit["confidence"] = compute_confidence(hit, text)

    # Step 6: convert to PIIEntity
    entities: list[PIIEntity] = []
    for hit in merged:
        entities.append(PIIEntity(
            entity_type=hit["type"],
            text=hit["text"],
            start=hit["start"],
            end=hit["end"],
            score=hit["confidence"],
            page_num=page_num,
            source=hit.get("source", "regex"),
            analysis=f"{hit.get('source', 'regex')} | {hit['type']}",
        ))

    return entities


def detect_pii_all_pages(pages: list, **kwargs) -> list[PIIEntity]:
    """Run detection across multiple PageText objects in parallel."""
    all_entities: list[PIIEntity] = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(detect_pii, page.text, page.page_num, **kwargs): page.page_num
            for page in pages
        }
        for future in as_completed(futures):
            all_entities.extend(future.result())

    all_entities.sort(key=lambda e: (e.page_num, e.start))
    return all_entities


# ---------------------------------------------------------------------------
# Deduplication / merge helpers
# ---------------------------------------------------------------------------

def _deduplicate(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Remove overlapping detections, keeping the one with the higher score."""
    if not entities:
        return entities
    entities.sort(key=lambda e: (e.start, -e.score))
    result: list[PIIEntity] = [entities[0]]
    for ent in entities[1:]:
        prev = result[-1]
        if ent.start >= prev.end:
            result.append(ent)
        elif ent.score > prev.score:
            result[-1] = ent
    return result


def merge_entity_lists(*lists: list[PIIEntity]) -> list[PIIEntity]:
    """Merge multiple entity lists, deduplicating overlaps per page.

    When entities from different sources (LLM, NER) overlap, the one with the
    higher confidence score is kept.  The result is sorted by page then offset.
    """
    by_page: dict[int, list[PIIEntity]] = {}
    for entity_list in lists:
        for ent in entity_list:
            by_page.setdefault(ent.page_num, []).append(ent)

    merged: list[PIIEntity] = []
    for page_num in sorted(by_page):
        merged.extend(_deduplicate(by_page[page_num]))
    return merged
