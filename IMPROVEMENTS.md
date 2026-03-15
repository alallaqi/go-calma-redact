# GoCalma Redact — Improvement Backlog

This file tracks known improvements and future feature ideas to circle back on.
Items are grouped by area and roughly ordered by impact.

---

## NLP / PII Detection

### Multilingual NLP engine support
**Status:** Not started
**Priority:** High
The current Presidio setup hardcodes `lang_code: "en"` even though the app targets Swiss/German
documents. Add German (`de`) and multilingual model options so entities like names and addresses
in German text are detected with higher recall.

Steps:
- Add `spacy/de_core_news_lg` and `spacy/xx_ent_wiki_sm` (multilingual) to `NLP_MODELS`
- Auto-detect language of each page (`langdetect` is already in the venv) and pass the correct
  `language=` parameter to `detect_pii`
- Add German address/postal patterns to `_SWISS_RECOGNIZERS`

---

### Smarter Swiss postal code matching
**Status:** Partially improved (lookahead added)
**Priority:** Medium
The `CH_POSTAL` pattern (`\b[1-9]\d{3}(?=\s+[A-ZÄÖÜ][a-zäöüß])`) still flags some
non-postal 4-digit numbers. Improve by:
- Bundling a static list of the ~3 500 Swiss postal codes for exact matching
- Raising the confidence score for exact matches

---

### LLM page-text chunking
**Status:** Not started
**Priority:** Medium
`llm_detect.py` truncates page text to 3 000 characters. Dense legal / medical pages can
be longer. Implement chunking: split the page into overlapping windows that each fit in the
model's context, run verification on each window, and merge results de-duplicating overlaps.

---

### Multilingual LLM models
**Status:** Not started
**Priority:** Low
`LLM_MODELS` currently only lists Mistral-7B. Add multilingual options:
- `Mistral-7B-Instruct-v0.3` (already there, handles German reasonably)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` for higher accuracy
- Remote API option (Claude / GPT-4) with a clear privacy warning that data leaves the machine

---

## Performance

### PDF document cache
**Status:** Not started
**Priority:** Medium
Each call to `render_page`, `extract_words`, `map_words_to_entities`, etc. independently
opens and parses the same PDF bytes. A simple `functools.lru_cache` keyed on `hash(pdf_bytes)`
would eliminate repeated I/O on the hot path during interactive review.

Implementation sketch:
```python
import functools, hashlib, fitz

@functools.lru_cache(maxsize=4)
def _open_doc(pdf_hash: str, pdf_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=pdf_bytes, filetype="pdf")
```
The `pdf_hash` key prevents stale cache hits; `maxsize=4` keeps at most 4 PDFs in memory.

---

### Async / background NER for large documents
**Status:** Not started
**Priority:** Low
For documents >20 pages, run NER in a background thread and stream results page-by-page
into the UI using `st.empty()` placeholders, so the user sees partial results immediately
rather than waiting for all pages to finish.

---

## Security

### Mypy strict typing in CI
**Status:** Not started
**Priority:** Low
All source files use `from __future__ import annotations`. Adding `mypy --strict` to a
GitHub Actions workflow would catch type errors before they reach production.

Config (`pyproject.toml`):
```toml
[tool.mypy]
strict = true
python_version = "3.10"
```

---

### Rate-limit / size guard for OCR
**Status:** Partially done (DPI lowered to 200, file size capped at 50 MB)
**Priority:** Low
For very long scanned PDFs the sequential OCR loop still holds each rendered pixmap in memory
until the page is processed. Explicitly `del pix` after `image_to_string` returns to release
memory immediately:
```python
pix = page.get_pixmap(dpi=dpi)
png_bytes = pix.tobytes("png")
del pix   # release ~6 MB immediately
img = Image.open(io.BytesIO(png_bytes))
```

---

## UX / Features

### Undo / redo for manual word toggles
**Status:** Not started
**Priority:** Medium
The interactive viewer lets users double-click to toggle individual words, but there is no
undo. Store a toggle history in session state (`st.session_state.toggle_history`) and add
an "Undo last change" button.

---

### Batch / CLI mode
**Status:** Not started
**Priority:** Medium
Add a command-line interface so GoCalma can be used in automated pipelines:
```
python -m gocalma redact input.pdf --approach replace --output redacted.pdf --key-file key.gocalma
```
The `gocalma/` module already exposes all the building blocks; a thin `__main__.py` would
wire them together without touching the Streamlit UI.

---

### Confidence threshold slider in UI
**Status:** Not started
**Priority:** Low
Expose the `score_threshold` parameter (currently hardcoded to 0.35) as a sidebar slider
so users can tune sensitivity without editing code.

---

### Export review report
**Status:** Not started
**Priority:** Low
Allow users to download a CSV / JSON report of all detected entities (type, text snippet,
page, confidence score, approved/rejected) alongside the redacted PDF for audit purposes.

---

## Developer Experience

### CI / CD pipeline
**Status:** Not started
**Priority:** Medium
Add a GitHub Actions workflow that:
1. Installs dependencies
2. Runs `pytest tests/`
3. Runs `mypy --strict gocalma/`
4. Uploads coverage report

---

### Docker image
**Status:** Not started
**Priority:** Low
Provide a `Dockerfile` that pre-installs all dependencies (including the spaCy model) so
users can run GoCalma with a single `docker run` without managing a Python environment.

---

### Changelog
**Status:** Not started
**Priority:** Low
Add a `CHANGELOG.md` following Keep a Changelog conventions to track versions and changes
for external users.
