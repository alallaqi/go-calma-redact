"""PII detection using Microsoft Presidio with configurable NLP backends."""

from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider


# None = detect ALL entity types Presidio knows about (no US-only filter)
ENTITIES = None

# Maximum number of NLP engines to keep in memory simultaneously.
# Each spaCy/flair model can consume 300 MB – 1.5 GB; cap at 2 to avoid OOM.
_MAX_ENGINES = 2

# Custom recognizers for Swiss / European PII that Presidio doesn't ship with.
# Presidio only fires a PatternRecognizer when analyze() is called with a
# matching language, so we create one instance per supported language.
def _make_swiss_recognizers(lang: str) -> list[PatternRecognizer]:
    return [
        PatternRecognizer(
            supported_entity="CH_AHV",
            name="Swiss AHV Number",
            patterns=[
                # Standard format: 756.1234.5678.90
                Pattern("AHV_DOTS", r"\b756\.\d{4}\.\d{4}\.\d{2}\b", 0.95),
                # OCR-tolerant: dots may be rendered as spaces or commas
                Pattern("AHV_SPACE", r"\b756[\.\s,]\d{4}[\.\s,]\d{4}[\.\s,]\d{2}\b", 0.80),
            ],
            context=["ahv", "ahvn", "ahvn13", "avs", "n13",
                     "sozialversicherung", "insurance", "versicherung"],
            supported_language=lang,
        ),
        PatternRecognizer(
            supported_entity="CH_POSTAL",
            name="Swiss Postal Code",
            # Lookahead requires a capitalised word (city name) after the code;
            # this drastically reduces false positives on years, prices, etc.
            patterns=[Pattern("CH_ZIP", r"\b[1-9]\d{3}(?=\s+[A-ZÄÖÜ][a-zäöüß])", 0.5)],
            context=["strasse", "weg", "platz", "gasse", "switzerland", "schweiz", "suisse"],
            supported_language=lang,
        ),
        PatternRecognizer(
            supported_entity="ADDRESS",
            name="Street Address (DACH)",
            patterns=[
                Pattern("street_de", r"\b[A-ZÄÖÜ][a-zäöüß]+(?:strasse|weg|gasse|platz|allee|ring)\s+\d+\b", 0.7),
                Pattern("street_num", r"\b[A-ZÄÖÜ][a-zäöüß]+(?:strasse|weg|gasse|platz|allee|ring)\s+\d+[a-z]?\b", 0.7),
            ],
            supported_language=lang,
        ),
        # German-language date format: "26. März 1975", "31. Mai 2026"
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="German Date",
            patterns=[Pattern(
                "DE_DATE",
                r"\b\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August"
                r"|September|Oktober|November|Dezember)\s+\d{4}\b",
                0.85,
            )],
            supported_language=lang,
        ),
        # Swiss phone numbers: "044 123 45 67", "044 123 45 67"
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="Swiss Phone (local format)",
            patterns=[
                Pattern("CH_PHONE_3", r"\b0\d{2}\s\d{3}\s\d{2}\s\d{2}\b", 0.75),
                Pattern("CH_PHONE_2", r"\b0\d{2}\s\d{3}\s\d{4}\b", 0.70),
            ],
            context=["tel", "telefon", "mobile", "fax", "natel", "phone", "kontakt"],
            supported_language=lang,
        ),
        # Swiss access / login codes: "ABCD-EFgh-IJKL-MNop", "9LT5-HT5H-TW", "BCLX G0FR"
        # Mixed case allowed — many Swiss cantonal portals use camelCase segments.
        PatternRecognizer(
            supported_entity="CH_ACCESS_CODE",
            name="Swiss Access / Login Code",
            patterns=[
                # 3–5 hyphen-separated segments of 2–8 chars (covers 4-4-4-4, 4-4-2, etc.)
                Pattern("CH_CODE_DASH", r"\b[A-Za-z0-9]{2,8}(?:-[A-Za-z0-9]{2,8}){2,4}\b", 0.75),
                # Space-separated 4+4 uppercase blocks: "BCLX G0FR"
                Pattern("CH_CODE_SPACE", r"\b[A-Z][A-Z0-9]{3}\s[A-Z0-9]{4}\b", 0.60),
            ],
            context=["zugangscode", "code", "zugang", "access", "login", "passwort",
                     "password", "pin", "kennung", "referenz", "deklaration"],
            supported_language=lang,
        ),
        # Swiss personal / document reference IDs: "12-3456-78", "100000000000"
        PatternRecognizer(
            supported_entity="CH_ID_NUMBER",
            name="Swiss Personal / Document ID",
            patterns=[
                # YY-NNNN-NN style (Kanton Zürich PID format)
                Pattern("CH_PID", r"\b\d{2}-\d{4}-\d{2}\b", 0.75),
                # Long numeric document IDs (10-13 digits)
                Pattern("CH_DOC_ID", r"\b\d{10,13}\b", 0.55),
            ],
            context=["pid", "persönliche id", "persönlich", "dokument", "dokumenten",
                     "id", "identifikation", "deklaration", "steuernummer"],
            supported_language=lang,
        ),
    ]


# Maps each backend engine name to the Python package it requires.
_BACKEND_PACKAGES: dict[str, str] = {
    "spacy": "spacy",
    "flair": "flair",
    "transformers": "transformers",
    "stanza": "stanza",
}

NLP_MODELS: dict[str, dict] = {
    # ---------- English models ----------
    "spaCy/en_core_web_lg": {
        "engine_name": "spacy",
        "model_name": "en_core_web_lg",
        "lang_codes": ["en"],
    },
    "flair/ner-english-large": {
        "engine_name": "flair",
        "model_name": "flair/ner-english-large",
        "lang_codes": ["en"],
    },
    "HuggingFace/obi/deid_roberta_i2b2": {
        "engine_name": "transformers",
        # TransformersNlpEngine requires a dict with both "spacy" (for tokenisation)
        # and "transformers" (for NER) keys.
        "model_name": {
            "spacy": "en_core_web_sm",
            "transformers": "obi/deid_roberta_i2b2",
        },
        "lang_codes": ["en"],
    },
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base": {
        "engine_name": "transformers",
        "model_name": {
            "spacy": "en_core_web_sm",
            "transformers": "StanfordAIMI/stanford-deidentifier-base",
        },
        "lang_codes": ["en"],
    },
    "HuggingFace/dslim/bert-base-NER": {
        "engine_name": "transformers",
        "model_name": {
            "spacy": "en_core_web_sm",
            "transformers": "dslim/bert-base-NER",
        },
        "lang_codes": ["en"],
    },
    "stanza/en": {
        "engine_name": "stanza",
        "model_name": "en",
        "lang_codes": ["en"],
    },
    # ---------- Swiss / multilingual models ----------
    "HuggingFace/ZurichNLP/swissbert-ner": {
        "engine_name": "transformers",
        "model_name": {
            "spacy": "en_core_web_sm",
            "transformers": "ZurichNLP/swissbert-ner",
        },
        "lang_codes": ["de", "fr", "it", "rm"],
        "swissbert": True,   # triggers XMod language adapter init
    },
    # German spaCy model — install: python -m spacy download de_core_news_lg
    "spaCy/de_core_news_lg": {
        "engine_name": "spacy",
        "model_name": "de_core_news_lg",
        "lang_codes": ["de"],
    },
    # Multilingual spaCy (DE/FR/IT/EN) — install: python -m spacy download xx_ent_wiki_sm
    "spaCy/xx_ent_wiki_sm": {
        "engine_name": "spacy",
        "model_name": "xx_ent_wiki_sm",
        "lang_codes": ["de", "fr", "it", "en"],
    },
}

DEFAULT_MODEL = "spaCy/en_core_web_lg"


def _backend_installed(engine_name: str) -> bool:
    """Check whether the Python package for a backend is importable."""
    pkg = _BACKEND_PACKAGES.get(engine_name)
    if pkg is None:
        return False
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def available_models() -> dict[str, dict]:
    """Return the subset of NLP_MODELS whose backend packages are installed."""
    return {
        key: cfg
        for key, cfg in NLP_MODELS.items()
        if _backend_installed(cfg["engine_name"])
    }


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
# Engine cache — bounded LRU, thread-safe
# ---------------------------------------------------------------------------

_engines: dict[str, AnalyzerEngine] = {}
_engine_order: list[str] = []   # tracks insertion order for LRU eviction
_engines_lock = threading.Lock()


def _build_swissbert_engine(cfg: dict, lang_codes: list[str]) -> AnalyzerEngine:
    """Build a Presidio AnalyzerEngine backed by SwissBERT-NER.

    SwissBERT is an XMod model that requires ``set_default_language`` to be
    called after the pipeline is loaded, otherwise it produces no NER output.
    We call it here immediately after provider.create_engine() returns.
    """
    from presidio_analyzer.nlp_engine import TransformersNlpEngine  # noqa: F401

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "transformers",
        "models": [
            {"lang_code": lc, "model_name": cfg["model_name"]}
            for lc in lang_codes
        ],
    })
    nlp_engine = provider.create_engine()

    # Activate the language adapter inside the XMod model.
    try:
        for nlp_pipeline in nlp_engine.nlp.values():
            hf_pipe = nlp_pipeline.get_pipe("hf_token_pipe")
            model = hf_pipe.hf_pipeline.model
            if hasattr(model, "set_default_language"):
                model.set_default_language("de_CH")
    except Exception:
        pass  # If the structure changes in a future version, degrade gracefully.

    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=lang_codes)


def _get_engine(model_key: str = DEFAULT_MODEL) -> AnalyzerEngine:
    if model_key not in _engines:
        with _engines_lock:
            # Double-checked: another thread may have loaded it while we waited.
            if model_key not in _engines:
                # Evict the oldest engine if we are at the memory cap.
                if len(_engines) >= _MAX_ENGINES:
                    oldest = _engine_order.pop(0)
                    del _engines[oldest]

                cfg = NLP_MODELS[model_key]
                lang_codes = cfg["lang_codes"]

                if cfg.get("swissbert"):
                    engine = _build_swissbert_engine(cfg, lang_codes)
                else:
                    provider = NlpEngineProvider(nlp_configuration={
                        "nlp_engine_name": cfg["engine_name"],
                        "models": [
                            {"lang_code": lc, "model_name": cfg["model_name"]}
                            for lc in lang_codes
                        ],
                    })
                    nlp_engine = provider.create_engine()
                    engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=lang_codes)

                for lc in lang_codes:
                    for rec in _make_swiss_recognizers(lc):
                        engine.registry.add_recognizer(rec)

                _engines[model_key] = engine
                _engine_order.append(model_key)

    return _engines[model_key]


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _detect_language(text: str, model_key: str) -> str:
    """Detect the language of *text* and return the best matching lang code.

    Falls back to the model's primary language if detection fails or the
    detected language is not supported by the selected model.
    """
    supported = NLP_MODELS[model_key]["lang_codes"]
    fallback   = supported[0]

    if not text or len(text.strip()) < 20:
        return fallback

    try:
        from langdetect import detect, DetectorFactory, LangDetectException
        DetectorFactory.seed = 42  # deterministic
        detected = detect(text[:2000])
        # langdetect returns BCP-47 codes like "de", "fr", "it", "en"
        return detected if detected in supported else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_pii(
    text: str,
    page_num: int,
    language: str | None = None,
    score_threshold: float = 0.35,
    model_key: str = DEFAULT_MODEL,
) -> list[PIIEntity]:
    """Run Presidio analysis on *text* and return detected PII above the threshold.

    *language* defaults to auto-detected language of the text (capped to what
    the selected model supports).  Pass an explicit BCP-47 code to override.
    """
    if language is None:
        language = _detect_language(text, model_key)
    engine = _get_engine(model_key)
    analyze_kwargs: dict = {
        "text": text,
        "language": language,
        "score_threshold": score_threshold,
    }
    if ENTITIES is not None:
        analyze_kwargs["entities"] = ENTITIES
    results: list[RecognizerResult] = engine.analyze(**analyze_kwargs)

    entities: list[PIIEntity] = []
    for r in results:
        explanation = ""
        if r.analysis_explanation:
            recognizer = r.analysis_explanation.recognizer or ""
            pattern = r.analysis_explanation.pattern_name or ""
            parts = [p for p in [recognizer, pattern] if p]
            explanation = " | ".join(parts) if parts else "Presidio"

        entities.append(
            PIIEntity(
                entity_type=r.entity_type,
                text=text[r.start : r.end],
                start=r.start,
                end=r.end,
                score=float(r.score),  # cast from numpy float32 to Python float
                page_num=page_num,
                analysis=explanation or "NER detection",
            )
        )

    return _deduplicate(entities)


def detect_pii_all_pages(pages: list, **kwargs) -> list[PIIEntity]:
    """Run detection across multiple PageText objects in parallel.

    Pages are processed concurrently using a thread pool.  The engine itself
    is loaded once (protected by ``_engines_lock``) and reused across threads
    because ``AnalyzerEngine.analyze`` is thread-safe after initialisation.
    """
    all_entities: list[PIIEntity] = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(detect_pii, page.text, page.page_num, **kwargs): page.page_num
            for page in pages
        }
        for future in as_completed(futures):
            all_entities.extend(future.result())

    # Sort for deterministic output order (page then character position).
    all_entities.sort(key=lambda e: (e.page_num, e.start))
    return all_entities


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
