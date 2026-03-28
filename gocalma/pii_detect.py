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

_NER_MODEL_ID = "Davlan/bert-base-multilingual-cased-ner-hrl"
_NER_LABEL_MAP: dict[str, str] = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE_TIME",
}

_ner_pipe = None
_ner_pipe_lock = threading.Lock()


def _get_ner_pipe():
    """Lazily load the multilingual NER pipeline (thread-safe singleton)."""
    global _ner_pipe
    if _ner_pipe is None:
        with _ner_pipe_lock:
            if _ner_pipe is None:
                from transformers import pipeline
                _ner_pipe = pipeline(
                    "ner",
                    model=_NER_MODEL_ID,
                    aggregation_strategy="simple",
                )
    return _ner_pipe


# Backward-compat exports — other modules import these names.
DEFAULT_MODEL = _NER_MODEL_ID
NLP_MODELS: dict[str, dict] = {
    _NER_MODEL_ID: {
        "engine_name": "transformers",
        "model_name": _NER_MODEL_ID,
        "lang_codes": ["de", "fr", "it", "en", "es", "pt", "nl"],
    },
}


def available_models() -> dict[str, dict]:
    """Return available NER models (always the single multilingual model)."""
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

    # Step 6: convert to PIIEntity
    entities: list[PIIEntity] = []
    for hit in merged:
        entities.append(PIIEntity(
            entity_type=hit["type"],
            text=hit["text"],
            start=hit["start"],
            end=hit["end"],
            score=hit.get("priority", 5) / 10.0,  # normalise priority to 0-1 score
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
