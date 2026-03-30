# Contributing to GoCalma Redact

Thanks for your interest in contributing. This guide covers how to set up the project, run tests, and submit changes.

## Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDF tests)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (optional, for containerised development)
- [Ollama](https://ollama.ai) (optional, for LLM verification features)

## Development Setup

```bash
git clone https://github.com/alallaqi/go-calma-redact
cd gocalma-redact
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract for OCR tests:

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-ita
```

## Running Tests

```bash
# Full suite (185 tests)
python -m pytest tests/ -v

# Single module
python -m pytest tests/test_crypto.py -v

# Stop on first failure
python -m pytest tests/ -x
```

All tests run without a GPU, Ollama, or network access. The NER model downloads on first run (~680 MB) and is cached in `~/.cache/huggingface`.

## Running the App Locally

```bash
streamlit run app.py
# → http://localhost:8501
```

Or with Docker:

```bash
docker compose up --build
```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of modules, the detection pipeline, and design decisions.

Key modules:

| Module | Purpose |
|--------|---------|
| `regex_patterns.py` | 35 compiled PII patterns (Swiss, EU, universal) |
| `pii_detect.py` | Chunked BERT NER + confidence scoring |
| `llm_detect.py` | Ollama LLM verification (additive only) |
| `redactor.py` | 7 de-identification modes |
| `crypto.py` | AES-256-GCM encryption for key files |
| `audit.py` | GDPR audit trail (metadata only) |
| `pdf_extract.py` | Text extraction + OCR fallback |

## How to Contribute

### Reporting Bugs

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, and whether you're using Docker

### Adding a New Regex Pattern

1. Add the pattern to the appropriate list in `regex_patterns.py` (`PATTERNS`, `LABEL_CONTEXT_PATTERNS`, or `HEALTH_PATTERNS`)
2. Assign a priority following the existing scale (see `ARCHITECTURE.md`)
3. Add a test in `tests/test_recognizers.py` or `tests/test_pii_detect.py`
4. Run the full suite to check for regressions

### Adding a New Entity Type

1. Add the regex pattern (if applicable) to `regex_patterns.py`
2. Update `_default_ner_priority()` in `regex_patterns.py` if the type comes from NER
3. Add the type to `CRITICAL_TYPES` in `regex_patterns.py` if it's high-sensitivity
4. Update the LLM prompt templates in `llm_detect.py` (both `_VERIFY_PROMPT_TEMPLATE` and `_DETECT_PROMPT_TEMPLATE`)
5. Add to `_NEVER_DISPUTE` or `_ALWAYS_PROTECTED_TYPES` in `llm_detect.py` if the LLM should never dispute it
6. Add tests

### Adding a New Redaction Mode

1. Add the approach name to `APPROACHES` in `redactor.py`
2. Implement the text replacement logic in the `_replacement_text()` function
3. Add tests in `tests/test_redactor.py`
4. The UI picks up new approaches automatically from the `APPROACHES` list

### Modifying the Encryption Format

If you change the `.gocalma` file format:
- Add a new sentinel (e.g. `_SENTINEL_V4`)
- Keep backward-compatible reading of all previous versions in `load_key_file()`
- Never break existing `.gocalma` files — users depend on them to restore redacted documents
- Update tests in `tests/test_crypto.py` including a legacy compatibility test

## Code Style

- No enforced formatter — match the style of surrounding code
- Type hints on public function signatures
- Docstrings on all public functions (one-liner for simple functions, NumPy-style for complex ones)
- No unnecessary abstractions — three similar lines of code is better than a premature helper

## Testing Guidelines

- All new features need tests
- Tests must pass without network access, GPU, or Ollama
- Use `pytest.raises` for expected exceptions
- Test files mirror module names: `gocalma/crypto.py` → `tests/test_crypto.py`

## Security

If you discover a security vulnerability, please do **not** open a public issue. Instead, email the maintainers directly.

Key security invariants to preserve:
- Regex-sourced entities can never be disputed by the LLM
- Protected types (`_NEVER_DISPUTE`) can never be marked as false positives
- The audit log never contains document content
- Encryption key files use authenticated encryption (AES-256-GCM)
- Document content in LLM prompts is wrapped in `<document_content>` delimiters

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `python -m pytest tests/ -v` — all tests must pass
4. Write a clear PR description explaining what and why
5. Keep PRs focused — one feature or fix per PR
