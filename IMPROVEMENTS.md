# GoCalma Redact — Improvements & Status

This file tracks implemented improvements, current capabilities, and future ideas.

---

## Completed

### Multilingual NER Detection
**Status:** Done
Replaced single-language Presidio/spaCy with a multilingual BERT model (`Davlan/bert-base-multilingual-cased-ner-hrl`) that handles EN, DE, FR, IT, ES, PT, and NL natively. Two additional models selectable in the UI: `dslim/bert-base-NER` (English-only, faster) and `Davlan/xlm-roberta-large-ner-hrl` (highest accuracy). Language is auto-detected per page via `langdetect`.

### Chunked NER Inference
**Status:** Done
BERT NER now uses chunked inference with 460-token windows and 50-token overlap, eliminating the previous 4,500-character truncation limit. Documents of any length are processed correctly. Entities from overlap zones are deduplicated by span overlap and text identity.

### Swiss-Native Regex Patterns (35 patterns across 3 tiers)
**Status:** Done
- **24 core patterns:** AHV/AVS (dot + space formats), Swiss IBAN, Zugangscode, CH postal code, CH personal ID, CH reference ID, insurance number (3-3-3), US SSN, UK NI, German Steuer-ID, French NIR, Italian Codice Fiscale, Spanish DNI/NIE, ICAO passport, email, international phone (E.164), any-country IBAN, credit card (Luhn-validated), IPv4, date of birth
- **6 label-context patterns:** Detect PII by surrounding keywords — "Name:", "Herr"/"Frau", "Versicherungs-Nr.", "Geburtsdatum", "Pass-Nr.", "Adresse:"
- **5 health/medical patterns:** ICD-10 codes, diagnosis context, medication context, allergy context, blood type

### Credit Card Luhn Validation
**Status:** Done
Credit card regex now validates the Luhn checksum, eliminating false positives from arbitrary digit sequences.

### AES-256-GCM Encryption
**Status:** Done
Replaced Fernet (AES-128-CBC + HMAC-SHA256) with AES-256-GCM authenticated encryption. Key files use `.gocalma` v3 format with 96-bit random nonces and 32-byte keys. Password-protected key files use PBKDF2-HMAC-SHA256 with 480,000 iterations and random 16-byte salt. Backward-compatible reading of legacy v1/v2 Fernet files preserved.

### LLM Verification — Additive Only, Batch Mode
**Status:** Done
Optional local LLM (any Ollama model) acts as a second pass. The LLM can never dispute protected entity types (PERSON, AHV, IBAN, SSN, credit card, DOB) or regex-sourced entities. Uses **batch verification** — all non-protected entities across every page sent in a single LLM call (up to 8,000 chars combined), reducing latency proportional to page count. Falls back silently if Ollama is unavailable.

### Document-Type Classification
**Status:** Done
LLM classifies documents as insurance, medical, police, tax, government, or general. Domain-specific context is injected into verification prompts to improve recall for sector-specific PII (policy numbers, patient IDs, case numbers, etc.).

### Context-Aware False Positive Filters
**Status:** Done
Automatic filters for: country names in product descriptions, abbreviation table entries, premium region codes, generic street names without house numbers. Computed confidence scores replace raw NER token probabilities, incorporating source type, entity type floors, span length, context keywords, and repetition signals.

### Docker One-Command Setup
**Status:** Done
Complete Docker deployment with pre-baked models:
- `Dockerfile` — Python 3.11-slim with Tesseract OCR (DE/FR/IT/EN), multilingual BERT NER (~680 MB), and Flan-T5 summariser (~77 MB) pre-downloaded at build time. First run is instant.
- `docker-compose.yml` — Default profile (NER + regex only) and `ollama` profile that adds an Ollama sidecar with auto-pull of `qwen2.5:1.5b`.
- `start.sh` / `start.bat` — Interactive launcher that detects Docker, offers NER-only vs LLM mode, and falls back to manual instructions.

```bash
./start.sh          # interactive launcher
# or
docker compose up   # NER only
docker compose --profile ollama up  # with LLM
```

### Hugging Face Spaces Deployment
**Status:** Live at [huggingface.co/spaces/al-allaqi/gocalma-redact](https://huggingface.co/spaces/al-allaqi/gocalma-redact)
Cloud demo running on HF Spaces (Docker SDK). Includes NER + regex detection with a cloud privacy notice banner. LLM verification is local-only (requires Ollama). The HF deployment uses a separate Dockerfile optimised for the Spaces environment (non-root user, port 7860, CPU-only PyTorch).

### OCR Pipeline
**Status:** Done
Automatic OCR for scanned/image-only PDFs via Surya (transformer-based, preferred) or Tesseract (fallback). Word bounding boxes stored with exact character offsets for precise redaction. Docker image includes Tesseract with DE/FR/IT/EN language packs.

### 7 De-identification Modes
**Status:** Done
Redact (black box), replace (`<PERSON>`), mask (`****`), hash (salted HMAC-SHA256), encrypt (AES-256-GCM label), highlight (yellow), synthesize (fake data). All modes are reversible via the encrypted `.gocalma` key file.

### GDPR Audit Trail
**Status:** Done
Timestamped JSON audit log per redaction recording entity type counts, severity classification, redaction mode, and model used. No document content is ever stored in the audit log.

### Comprehensive Test Suite
**Status:** Done — 185 tests across 6 test files
- `test_recognizers.py` — Regex, merge, confidence, LLM protection, false positives
- `test_llm_detect.py` — LLM prompt parsing, doc classification, batch verification
- `test_pii_detect.py` — Chunked NER, deduplication, language detection
- `test_pdf_extract.py` — PDF extraction, OCR, page limits
- `test_redactor.py` — All 7 de-id modes, reversibility, HMAC hash
- `test_crypto.py` — AES-256-GCM encryption, PBKDF2, key file formats, legacy compat

### Documentation
**Status:** Done
- `README.md` — Full feature documentation, detection pipeline, security model, quick start
- `ARCHITECTURE.md` — Module dependency graph, pipeline diagrams, design decisions
- `CONTRIBUTING.md` — Dev setup, how-to guides, security invariants, PR guidelines

---

## Future Ideas

### Batch / CLI Mode
Add a command-line interface for automated pipelines:
```
python -m gocalma redact input.pdf --approach replace --output redacted.pdf
```

### CI / CD Pipeline
GitHub Actions workflow: pytest, mypy --strict, coverage report.

### React / Vanilla JS Frontend
Replace Streamlit with a standalone frontend for Vercel deployment.

### Confidence Threshold Slider
Expose `score_threshold` as a sidebar slider so users can tune sensitivity.

### Undo / Redo for Manual Word Toggles
Store toggle history in session state with an undo button.

### Swiss Postal Code Exact Matching
Bundle the ~3,500 Swiss postal codes for exact matching instead of regex-only.
