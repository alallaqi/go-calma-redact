"""GoCalma — Local PII Redaction Tool.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Log capture — surface warnings from detection modules in the Streamlit UI
# ---------------------------------------------------------------------------

class _WarningCollector(logging.Handler):
    """Collects WARNING+ log records from gocalma.* modules."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def flush_warnings(self) -> list[str]:
        msgs = [self.format(r) for r in self.records]
        self.records.clear()
        return msgs


_warning_collector = _WarningCollector()
_warning_collector.setFormatter(logging.Formatter("%(name)s: %(message)s"))
logging.getLogger("gocalma").addHandler(_warning_collector)

from gocalma.pdf_extract import extract_text
from gocalma.pii_detect import detect_pii_all_pages, PIIEntity, NLP_MODELS, DEFAULT_MODEL, available_models
from gocalma.llm_detect import llm_verify_all_pages, LLM_MODELS, is_llm_available

# HuggingFace repo IDs for each local transformers model (used in download hints).
_HF_IDS: dict[str, str] = {
    "Mistral-7B-Instruct (local)":   "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3.5-mini-Instruct (local)": "microsoft/Phi-3.5-mini-instruct",
    "Qwen2.5-1.5B-Instruct (local)": "Qwen/Qwen2.5-1.5B-Instruct",
}
from gocalma.redactor import (
    redact_pdf, render_page, render_preview, page_count,
    extract_words, map_words_to_entities,
    APPROACHES, DEFAULT_APPROACH,
)
from gocalma.components.pdf_viewer import pdf_viewer
from gocalma.crypto import (
    generate_key, encrypt_mapping, save_key_file, load_key_file, decrypt_mapping,
)

_MAX_PDF_MB = 50  # Upload size cap shown in the UI


def _sanitize_error(exc: Exception) -> str:
    """Return exception message with filesystem paths stripped out."""
    import re
    msg = f"{type(exc).__name__}: {exc}"
    # Remove absolute paths (Unix and Windows style)
    msg = re.sub(r"(/[^\s:]+/|[A-Z]:\\[^\s:]+\\)", "<path>/", msg)
    return msg

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


def _merge_entities(ner: list[PIIEntity], llm: list[PIIEntity]) -> list[PIIEntity]:
    """Merge NER and LLM entity lists, deduplicating overlapping ranges per page."""
    if not llm:
        return ner

    by_page: dict[int, list[PIIEntity]] = {}
    for e in ner + llm:
        by_page.setdefault(e.page_num, []).append(e)

    merged: list[PIIEntity] = []
    for _page, ents in sorted(by_page.items()):
        ents.sort(key=lambda e: (e.start, -e.score))
        deduped: list[PIIEntity] = []
        for ent in ents:
            if deduped and ent.start < deduped[-1].end:
                if ent.score > deduped[-1].score:
                    deduped[-1] = ent
            else:
                deduped.append(ent)
        merged.extend(deduped)

    return merged

# ---------------------------------------------------------------------------
# Brand palette (matched from gocalma.ch)
# ---------------------------------------------------------------------------
NAVY = "#2b3a4e"
NAVY_DARK = "#1e2d3d"
NAVY_LIGHT = "#3d5a80"
SLATE = "#6b7b8d"
BG_ALT = "#f5f7fa"
WHITE = "#ffffff"

ENTITY_COLORS = {
    "PERSON": (NAVY, WHITE),
    "EMAIL_ADDRESS": ("#3d5a80", WHITE),
    "PHONE_NUMBER": ("#4a7c96", WHITE),
    "LOCATION": ("#557a72", WHITE),
    "IBAN_CODE": ("#6b7b8d", WHITE),
    "CREDIT_CARD": ("#8b5a6b", WHITE),
    "DATE_TIME": ("#7a6b8d", WHITE),
    "IP_ADDRESS": ("#5a7b6b", WHITE),
    "US_SSN": ("#8d4b4b", WHITE),
    "US_BANK_NUMBER": ("#4a6b8d", WHITE),
    "US_DRIVER_LICENSE": ("#6b5a8d", WHITE),
    "US_PASSPORT": ("#8d6b4a", WHITE),
    "CUSTOM": ("#4a6b6b", WHITE),
}

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
FAVICON_PATH = Path(__file__).parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="GoCalma Redact",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🛡️",
    layout="wide",
)

st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY_DARK};
    }}
    section[data-testid="stSidebar"] {{
        color: {WHITE};
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown *,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] label {{
        color: {WHITE} !important;
    }}
    /* Force dark text inside ALL white-background controls in sidebar */
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section *,
    section[data-testid="stSidebar"] .stAlert,
    section[data-testid="stSidebar"] .stAlert * {{
        color: {NAVY_DARK} !important;
    }}
    /* Radio button option labels */
    section[data-testid="stSidebar"] [role="radiogroup"] label p,
    section[data-testid="stSidebar"] [role="radiogroup"] label span {{
        color: {WHITE} !important;
    }}
    /* Dropdown popover (portaled to body, outside sidebar) */
    [data-baseweb="menu"],
    [data-baseweb="menu"] * {{
        color: {NAVY_DARK} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15) !important;
    }}
    section[data-testid="stSidebar"] .stCaption {{
        color: rgba(255,255,255,0.6) !important;
    }}
    section[data-testid="stSidebar"] .stCaption * {{
        color: rgba(255,255,255,0.6) !important;
    }}

    /* ---- Sidebar logo container ---- */
    section[data-testid="stSidebar"] [data-testid="stImage"] {{
        background: {WHITE};
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.25rem;
    }}

    /* ---- Headers ---- */
    h1, h2, h3 {{
        color: {NAVY_DARK} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    h1 {{
        font-size: 2rem !important;
    }}

    /* ---- Primary buttons ---- */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background-color: {NAVY} !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        transition: background-color 0.2s ease;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background-color: {NAVY_LIGHT} !important;
    }}

    /* ---- Secondary / default buttons ---- */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        border: 1.5px solid {NAVY} !important;
        color: {NAVY} !important;
        border-radius: 24px !important;
        background: transparent !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: {BG_ALT} !important;
    }}

    /* ---- Download buttons ---- */
    .stDownloadButton > button {{
        border: 1.5px solid {NAVY} !important;
        color: {NAVY} !important;
        border-radius: 24px !important;
        background: {WHITE} !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }}
    .stDownloadButton > button:hover {{
        background-color: {BG_ALT} !important;
    }}

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {{
        border-radius: 12px;
    }}
    [data-testid="stFileUploader"] section {{
        border: 2px dashed rgba(43,58,78,0.25) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {NAVY} !important;
    }}

    /* ---- Progress bars ---- */
    .stProgress > div > div > div {{
        background-color: {NAVY} !important;
    }}

    /* ---- Alert boxes ---- */
    .stAlert {{
        border-radius: 10px !important;
    }}

    /* ---- Entity badge ---- */
    .entity-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}

    /* ---- Dividers ---- */
    hr {{
        border-color: rgba(43,58,78,0.1) !important;
    }}

    /* ---- Review table header ---- */
    .review-header {{
        font-size: 0.8rem;
        font-weight: 700;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid {NAVY};
        margin-bottom: 0.5rem;
    }}

    /* ---- Force images to full size, hide expand button ---- */
    [data-testid="stImage"] {{
        width: 100% !important;
    }}
    [data-testid="stImage"] img {{
        width: 100% !important;
        max-width: 100% !important;
    }}
    button[title="View fullscreen"] {{
        display: none !important;
    }}

    /* ---- PDF card labels ---- */
    .pdf-card-label {{
        font-size: 0.75rem;
        font-weight: 700;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }}

    /* ---- Footer tagline ---- */
    .footer-tag {{
        text-align: center;
        color: {SLATE};
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "pdf_bytes": None,
        "pages": None,
        "entities": None,
        "approved": None,
        "redacted_pdf": None,
        "key_blob": None,
        "step": "upload",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(str(LOGO_PATH), use_container_width=True)
    st.caption("Local PII redaction for PDFs")
    st.divider()

    st.subheader("NER model")
    installed = available_models()
    all_keys = list(NLP_MODELS.keys())
    selected_model = st.selectbox(
        "NER model package",
        options=all_keys,
        index=all_keys.index(DEFAULT_MODEL),
        format_func=lambda k: k if k in installed else f"{k}  (not installed)",
        key="ner_model",
    )
    if selected_model not in installed:
        engine_name = NLP_MODELS[selected_model]["engine_name"]
        if engine_name == "swissbert":
            st.warning(
                "SwissBERT-NER model weights not found in cache (~500 MB).\n\n"
                "**Download:**\n"
                "```\npython -c \""
                "from transformers import pipeline; "
                "pipeline('token-classification', model='ZurichNLP/swissbert-ner')"
                "\"\n```"
            )
        else:
            st.warning(
                f"Install the **{engine_name}** package to use this model:\n\n"
                f"`pip install {engine_name}`"
            )

    st.divider()

    st.subheader("LLM detector")
    llm_options = ["None"] + list(LLM_MODELS.keys())
    selected_llm = st.selectbox(
        "LLM detector",
        options=llm_options,
        index=0,
        key="llm_model",
    )
    if selected_llm != "None":
        cfg = LLM_MODELS[selected_llm]
        speed = cfg.get("speed", "")
        if is_llm_available(selected_llm):
            st.caption(
                f"Verifies NER findings + catches missed PII.  \n"
                f"{speed} — NER runs first for instant results."
            )
        elif cfg.get("backend") == "ollama":
            st.warning(
                f"Ollama is not running or **{cfg['model']}** is not pulled.\n\n"
                "**Setup:**\n"
                f"```\nbrew install ollama\nollama serve\nollama pull {cfg['model']}\n```"
            )
        else:
            hf_id = _HF_IDS.get(selected_llm, Path(cfg["path"]).name)
            st.warning(
                f"Model weights not found.\n\n"
                "**Download:**\n"
                f"```\nhuggingface-cli download {hf_id}"
                f" --local-dir {cfg['path']}\n```"
            )

    st.divider()

    st.subheader("De-identification approach")
    approach_keys = list(APPROACHES.keys())
    selected_approach = st.selectbox(
        "De-identification approach",
        options=approach_keys,
        index=approach_keys.index(DEFAULT_APPROACH),
        format_func=lambda k: k,
        key="deid_approach",
    )
    st.caption(APPROACHES[selected_approach])

    st.divider()

    st.subheader("Output mode")
    output_mode = st.radio(
        "Redaction mode",
        options=["Permanent (flattened)", "Reversible (annotations)"],
        index=0,
        key="output_mode",
        help=(
            "**Permanent:** text is destroyed in the PDF — need the .gocalma key file "
            "to know what was redacted.\n\n"
            "**Reversible:** annotations cover the text visually but the original "
            "content is preserved underneath. Remove annotations in any PDF editor "
            "to reveal the original."
        ),
    )
    if output_mode == "Reversible (annotations)":
        st.warning(
            "**Security notice:** Reversible mode adds annotation overlays that "
            "visually hide PII but do **not** destroy it. Anyone with a PDF editor "
            "can remove the annotations and see the original text. "
            "Use **Permanent** mode for sensitive documents."
        )

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload your PDF\n"
        "2. Text is extracted (+ OCR)\n"
        "3. PII is detected locally\n"
        "4. Review & approve findings\n"
        "5. Download redacted PDF\n"
    )
    st.divider()
    st.markdown("Everything runs **locally**.  \nNo data leaves your machine.")

    st.divider()
    st.subheader("Reverse Redaction")
    key_file = st.file_uploader("Upload key file (.gocalma)", type=["gocalma"], key="unredact_key")
    if key_file is not None:
        unredact_pw = st.text_input(
            "Passphrase (leave blank if none was set)",
            type="password",
            key="unredact_password",
            help="Enter the passphrase used when the key file was created, or leave blank.",
        )
        try:
            blob = key_file.read()
            password_arg = unredact_pw if unredact_pw else None
            key, ciphertext = load_key_file(blob, password=password_arg)
            mapping = decrypt_mapping(ciphertext, key)
            st.success(f"Decrypted {len(mapping)} redaction(s)")
            st.json(mapping)
        except Exception as exc:
            st.error(f"Failed to decrypt: {_sanitize_error(exc)}")

# ---------------------------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------------------------

st.header("Upload your document")
st.markdown(f"<p style='color:{SLATE};margin-top:-0.5rem'>Drop a PDF below to get started. Sensitive information will be detected automatically.</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload", label_visibility="collapsed")

if uploaded is not None and st.session_state.pdf_bytes is None:
    raw = uploaded.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > _MAX_PDF_MB:
        st.error(
            f"File is {size_mb:.1f} MB — the limit is {_MAX_PDF_MB} MB. "
            "Please split the PDF into smaller parts and try again."
        )
    else:
        st.session_state.pdf_bytes = raw
        st.session_state.step = "extract"

# ---------------------------------------------------------------------------
# Step 2: Extract + Detect
# ---------------------------------------------------------------------------

if st.session_state.step == "extract" and st.session_state.pdf_bytes:
    try:
        with st.spinner("Extracting text from PDF..."):
            pages = extract_text(st.session_state.pdf_bytes)
            st.session_state.pages = pages

        ocr_pages = sum(1 for p in pages if p.is_ocr)
        total = len(pages)
        st.info(f"Extracted text from **{total}** page(s) ({ocr_pages} via OCR).")

        model_key = st.session_state.get("ner_model", DEFAULT_MODEL)
        if model_key not in available_models():
            st.error(
                f"**{model_key}** is not installed. "
                "Please select an installed model from the sidebar."
            )
            st.stop()

        with st.spinner(f"Detecting PII with **{model_key}**..."):
            ner_entities = detect_pii_all_pages(pages, model_key=model_key)

        llm_key = st.session_state.get("llm_model", "None")
        if llm_key != "None" and is_llm_available(llm_key):
            st.info(f"NER found **{len(ner_entities)}** entities. Running LLM verification...")
            with st.spinner(f"Verifying with **{llm_key}** (this may take 30-60s)..."):
                entities = llm_verify_all_pages(pages, ner_entities, model_key=llm_key)
        else:
            entities = ner_entities

        disputed = [e for e in entities if "false positive" in e.analysis.lower()]
        llm_new = [e for e in entities if e.source == "LLM"]
        approved_defaults = [
            "false positive" not in e.analysis.lower()
            for e in entities
        ]

        st.session_state.entities = entities
        st.session_state.approved = approved_defaults

        # Surface any warnings from OCR / NER / LLM modules.
        detection_warnings = _warning_collector.flush_warnings()
        ocr_failed = sum(1 for p in pages if p.is_ocr and not p.text.strip())

        summary = f"Found **{len(entities)}** potential PII entities"
        if disputed or llm_new:
            summary += f" ({len(disputed)} disputed, {len(llm_new)} new from LLM)"
        st.success(summary + ".")

        if ocr_failed:
            st.warning(
                f"**{ocr_failed} page(s) returned no text** after OCR. "
                "PII on those pages could not be detected. "
                "Check that the document is not corrupted or password-protected."
            )
        if detection_warnings:
            with st.expander(f"Detection warnings ({len(detection_warnings)})", expanded=False):
                for w in detection_warnings:
                    st.caption(w)

        st.session_state.step = "review"
        st.rerun()

    except Exception as exc:
        st.session_state.step = "upload"
        st.error(
            f"Processing failed: {_sanitize_error(exc)}\n\n"
            "Please check that the selected NER model is fully installed, "
            "then try uploading the file again."
        )

# ---------------------------------------------------------------------------
# Step 3: Review (interactive live preview)
# ---------------------------------------------------------------------------

if st.session_state.step == "review":
    st.header("Review detected PII")
    st.markdown(
        f"<p style='color:{SLATE};margin-top:-0.5rem'>"
        "Toggle findings in the table or <b>double-click any word</b> on the original to add / remove redactions."
        "</p>",
        unsafe_allow_html=True,
    )

    entities: list[PIIEntity] = st.session_state.entities
    approved: list[bool] = st.session_state.approved
    approach = st.session_state.get("deid_approach", DEFAULT_APPROACH)
    pdf_bytes = st.session_state.pdf_bytes
    total_pages = page_count(pdf_bytes)

    # -- Page selector ------------------------------------------------------
    if total_pages > 1:
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1, key="preview_page"
        ) - 1
    else:
        current_page = 0

    # -- Compute word-level data for the interactive viewer -----------------
    pages_data = st.session_state.get("pages")
    words = extract_words(pdf_bytes, current_page, pages=pages_data)
    word_to_entity, redacted_indices = map_words_to_entities(
        pdf_bytes, current_page, words, entities, approved, pages=pages_data,
    )
    active_entities = [e for e, ok in zip(entities, approved) if ok]
    active_on_page = sum(1 for e in active_entities if e.page_num == current_page)

    image_b64 = base64.b64encode(render_page(pdf_bytes, current_page)).decode()

    # -- Side-by-side PDF cards (always visible at top) ---------------------
    col_orig, col_preview = st.columns(2)

    with col_orig:
        with st.container(border=True):
            st.markdown(
                "<div class='pdf-card-label'>Original — double-click to edit</div>",
                unsafe_allow_html=True,
            )
            click = pdf_viewer(
                image_base64=image_b64,
                words=words,
                redacted_indices=redacted_indices,
                approach=approach,
                key=f"pdfv_{current_page}",
            )

    with col_preview:
        with st.container(border=True):
            st.markdown(
                f"<div class='pdf-card-label'>Preview — {approach} · {active_on_page} redaction(s)</div>",
                unsafe_allow_html=True,
            )
            preview_png = render_preview(pdf_bytes, current_page, active_entities, approach, pages=pages_data)
            st.image(preview_png, use_container_width=True)

    # -- Handle double-click from the interactive viewer --------------------
    if click is not None:
        click_ts = click.get("ts", 0)
        if click_ts != st.session_state.get("_last_click_ts", 0):
            st.session_state._last_click_ts = click_ts
            w_idx = click["word_index"]

            if w_idx in word_to_entity:
                ent_idx = word_to_entity[w_idx]
                st.session_state.approved[ent_idx] = not st.session_state.approved[ent_idx]
            else:
                new_ent = PIIEntity(
                    entity_type="CUSTOM",
                    text=click["text"],
                    start=0,
                    end=len(click["text"]),
                    score=1.0,
                    page_num=current_page,
                )
                st.session_state.entities.append(new_ent)
                st.session_state.approved.append(True)

            st.rerun()

    # -- Entity table (below the cards) ------------------------------------
    st.divider()
    if not entities:
        st.warning("No PII detected in this document.")
    else:
        active_count = sum(approved)
        with st.expander(f"**Detected entities ({active_count} / {len(entities)} active)**", expanded=True):
            col_header = st.columns([0.3, 1.5, 1.8, 0.6, 1.8, 0.6, 0.7])
            col_header[0].markdown("<div class='review-header'>✓</div>", unsafe_allow_html=True)
            col_header[1].markdown("<div class='review-header'>Entity</div>", unsafe_allow_html=True)
            col_header[2].markdown("<div class='review-header'>Detected text</div>", unsafe_allow_html=True)
            col_header[3].markdown("<div class='review-header'>Source</div>", unsafe_allow_html=True)
            col_header[4].markdown("<div class='review-header'>Analysis</div>", unsafe_allow_html=True)
            col_header[5].markdown("<div class='review-header'>Page</div>", unsafe_allow_html=True)
            col_header[6].markdown("<div class='review-header'>Score</div>", unsafe_allow_html=True)

            for i, ent in enumerate(entities):
                cols = st.columns([0.3, 1.5, 1.8, 0.6, 1.8, 0.6, 0.7])
                approved[i] = cols[0].checkbox("", value=approved[i], key=f"chk_{i}", label_visibility="collapsed")
                bg, fg = ENTITY_COLORS.get(ent.entity_type, (SLATE, WHITE))
                cols[1].markdown(
                    f"<span class='entity-badge' style='background:{bg};color:{fg}'>{ent.entity_type}</span>",
                    unsafe_allow_html=True,
                )
                cols[2].code(ent.text)
                src = getattr(ent, "source", "NER")
                src_bg = "#3d5a80" if src == "NER" else "#8b5a6b"
                cols[3].markdown(
                    f"<span class='entity-badge' style='background:{src_bg};color:#fff'>{src}</span>",
                    unsafe_allow_html=True,
                )
                analysis_text = getattr(ent, "analysis", "")
                is_disputed = "false positive" in analysis_text.lower()
                analysis_color = "#c0392b" if is_disputed else SLATE
                cols[4].markdown(
                    f"<span style='font-size:0.75rem;color:{analysis_color}'>{analysis_text}</span>",
                    unsafe_allow_html=True,
                )
                cols[5].caption(f"Page {ent.page_num + 1}")
                cols[6].progress(ent.score, text=f"{ent.score:.0%}")

            st.session_state.approved = approved

    # -- Action buttons -----------------------------------------------------
    st.divider()
    with st.expander("Key file passphrase", expanded=True):
        st.text_input(
            "Protect the key file with a passphrase",
            type="password",
            key="key_password",
            help=(
                "The .gocalma key file is encrypted with this passphrase via "
                "PBKDF2-HMAC-SHA256 (480k iterations). Without it, anyone who "
                "obtains the file can decrypt your redaction mapping."
            ),
        )
        _pw = st.session_state.get("key_password", "")
        if not _pw:
            st.warning(
                "**No passphrase set.** The key file will contain an unprotected "
                "encryption key. Anyone with the file can reverse your redactions. "
                "Set a passphrase above for production use."
            )
        elif len(_pw.strip()) < 8:
            st.warning(
                "**Passphrase too short.** Use at least 8 characters for adequate security."
            )
        elif _pw != _pw.strip():
            st.warning("Passphrase has leading/trailing spaces — make sure this is intentional.")

    col_a, col_b = st.columns(2)
    if col_a.button("Start Over", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    _pw_value = st.session_state.get("key_password", "")
    _pw_too_short = bool(_pw_value and len(_pw_value.strip()) < 8)
    if col_b.button(
        "Apply Redactions", type="primary", use_container_width=True,
        disabled=not any(approved) or _pw_too_short,
    ):
        st.session_state.step = "redact"
        st.rerun()

# ---------------------------------------------------------------------------
# Step 4: Redact & Download
# ---------------------------------------------------------------------------

if st.session_state.step == "redact":
    try:
        entities = st.session_state.entities
        approved = st.session_state.approved
        selected = [ent for ent, ok in zip(entities, approved) if ok]

        approach = st.session_state.get("deid_approach", DEFAULT_APPROACH)
        is_flatten = st.session_state.get("output_mode", "Permanent (flattened)") == "Permanent (flattened)"
        mode_label = "permanent" if is_flatten else "reversible"
        with st.spinner(f"Applying **{approach}** ({mode_label}) to {len(selected)} entities..."):
            redacted_bytes, mapping = redact_pdf(
                st.session_state.pdf_bytes, selected,
                approach=approach,
                pages=st.session_state.get("pages"),
                flatten=is_flatten,
            )
        st.session_state.is_flatten = is_flatten

        key = generate_key()
        ciphertext = encrypt_mapping(mapping, key)
        key_blob = save_key_file(key, ciphertext, password=st.session_state.get("key_password") or None)

        st.session_state.redacted_pdf = redacted_bytes
        st.session_state.key_blob = key_blob
        st.session_state.step = "done"
        st.rerun()

    except Exception as exc:
        st.session_state.step = "review"
        st.error(
            f"Redaction failed: {_sanitize_error(exc)}\n\n"
            "Your document and detected entities are still available. "
            "You can adjust the approach or entity selection and try again."
        )

if st.session_state.step == "done":
    st.header("Redaction complete")
    st.success("Your redacted PDF is ready for download.")

    pdf_bytes = st.session_state.pdf_bytes
    redacted_bytes = st.session_state.redacted_pdf
    total_pages = page_count(pdf_bytes)

    if total_pages > 1:
        done_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1, key="done_page"
        ) - 1
    else:
        done_page = 0

    col_orig, col_final = st.columns(2)
    with col_orig:
        with st.container(border=True):
            st.markdown("<div class='pdf-card-label'>Original</div>", unsafe_allow_html=True)
            st.image(render_page(pdf_bytes, done_page), use_container_width=True)

    with col_final:
        with st.container(border=True):
            st.markdown("<div class='pdf-card-label'>Redacted output</div>", unsafe_allow_html=True)
            st.image(render_page(redacted_bytes, done_page), use_container_width=True)

    st.divider()

    is_flatten = st.session_state.get("is_flatten", True)

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download Redacted PDF",
        data=redacted_bytes,
        file_name="redacted.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    col2.download_button(
        "Download Key File",
        data=st.session_state.key_blob,
        file_name="redaction_key.gocalma",
        mime="application/octet-stream",
        use_container_width=True,
    )

    if is_flatten:
        st.info(
            "**Permanent mode** — the text has been removed from the PDF. "
            "Keep the key file safe — it's the only way to know what was redacted."
        )
    else:
        st.info(
            "**Reversible mode** — the original text is preserved under the annotations. "
            "To reveal it, open the PDF in any editor (Preview, Acrobat, etc.) and delete "
            "the annotation overlays. The key file is also included for reference."
        )

    st.divider()
    if st.button("Start Over", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<div class='footer-tag'>GoCalma Redact &mdash; privacy, locally.</div>", unsafe_allow_html=True)
