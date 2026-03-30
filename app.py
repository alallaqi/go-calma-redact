"""GoCalma — Local PII Redaction Tool.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import logging
import os
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
from gocalma.pii_detect import detect_pii_all_pages, PIIEntity
from gocalma.llm_detect import (
    is_available as is_llm_available,
    verify_entities as llm_verify_entities,
    verify_entities_batch as llm_verify_entities_batch,
    classify_document,
    list_ollama_models, get_llm_model, set_llm_model,
)
from gocalma.summariser import build_summary
from gocalma.audit import create_audit, audit_to_bytes

from gocalma.redactor import (
    redact_pdf, render_page, render_preview, page_count,
    extract_words, map_words_to_entities,
    detect_redaction_mode, unredact_pdf,
    APPROACHES, DEFAULT_APPROACH,
)
from gocalma.components.pdf_viewer import pdf_viewer
from gocalma.crypto import (
    generate_key, encrypt_mapping, save_key_file, load_key_file, decrypt_mapping,
)

from ui.styles import inject_css, SLATE, NAVY_DARK, WHITE
from ui.review import render_risk_summary, render_entity_table

_MAX_PDF_MB = 50  # Upload size cap shown in the UI
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


def _sanitize_error(exc: Exception) -> str:
    """Return exception message with filesystem paths stripped out."""
    import re
    msg = f"{type(exc).__name__}: {exc}"
    msg = re.sub(r"(/[^\s:]+/|[A-Z]:\\[^\s:]+\\)", "<path>/", msg)
    return msg


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
# Page config & theme
# ---------------------------------------------------------------------------
FAVICON_PATH = Path(__file__).parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="GoCalma Redact",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🛡️",
    layout="wide",
)

# Cloud demo notice — only shown on Hugging Face Spaces
if os.environ.get("SPACE_ID"):
    st.info(
        "**Cloud demo** — your uploads are processed on this server and not stored, "
        "but for maximum privacy run GoCalma locally. "
        "[GitHub →](https://github.com/alallaqi/go-calma-redact)",
        icon="☁️",
    )

inject_css()


# ---------------------------------------------------------------------------
# Session state
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

    # -- Status bar --
    st.markdown("**NER:** Multilingual BERT (DE/FR/IT/ES/EN)")
    if is_llm_available():
        st.markdown(
            "<span style='color:#27ae60;font-weight:600'>LLM verification: Active &#10003;</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='color:rgba(255,255,255,0.45)'>LLM verification: Not available</span>",
            unsafe_allow_html=True,
        )
    st.divider()

    app_mode = st.radio(
        "Mode",
        options=["Redact", "De-redact"],
        index=0,
        key="app_mode",
        horizontal=True,
    )

    if app_mode == "Redact":
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
            "5. Download redacted PDF + audit log\n"
        )

    if app_mode == "De-redact":
        st.divider()
        st.markdown(
            "**How de-redaction works**\n\n"
            "Upload the redacted PDF and its `.gocalma` key file.\n\n"
            "**Reversible mode** — annotations are removed "
            "and the original PDF is restored for download.\n\n"
            "**Permanent mode** — the original text was destroyed. "
            "Only the redaction mapping (JSON) is shown for reference."
        )

    st.divider()
    st.markdown("Everything runs **locally**.  \nNo data leaves your machine.")

    # -- Advanced expander --
    with st.expander("Advanced", expanded=False):
        st.caption("Default settings work best for most documents.")
        st.divider()

        from gocalma.pii_detect import (
            NLP_MODELS, DEFAULT_MODEL, available_models,
            set_ner_model, get_ner_model,
        )
        st.markdown("**NER model**")
        installed = available_models()
        all_ner_keys = list(NLP_MODELS.keys())
        current_ner = get_ner_model()
        ner_idx = all_ner_keys.index(current_ner) if current_ner in all_ner_keys else 0
        selected_ner = st.selectbox(
            "NER model",
            options=all_ner_keys,
            index=ner_idx,
            format_func=lambda k: (
                NLP_MODELS[k].get("description", k)
                if k in installed
                else f"{NLP_MODELS[k].get('description', k)}  ⚠ not installed"
            ),
            key="ner_model_advanced",
            label_visibility="collapsed",
        )
        if selected_ner != current_ner:
            set_ner_model(selected_ner)

        st.divider()
        st.markdown("**LLM model** *(Ollama)*")
        if is_llm_available():
            ollama_models = list_ollama_models()
            current_llm = get_llm_model()
            if ollama_models:
                llm_idx = (
                    ollama_models.index(current_llm)
                    if current_llm in ollama_models
                    else 0
                )
                selected_llm = st.selectbox(
                    "LLM model",
                    options=ollama_models,
                    index=llm_idx,
                    key="llm_model_advanced",
                    label_visibility="collapsed",
                )
                if selected_llm != current_llm:
                    set_llm_model(selected_llm)
            else:
                st.info(f"Using **{current_llm}**")
        else:
            st.warning(
                "Ollama not running or no models pulled.\n\n"
                "```\nbrew install ollama\nollama serve\nollama pull qwen2.5:0.5b\n```"
            )


# ===========================================================================
# Main content
# ===========================================================================

if app_mode == "De-redact":
    # -----------------------------------------------------------------------
    # De-redact mode
    # -----------------------------------------------------------------------
    st.header("Reverse Redaction")
    st.markdown(
        f"<p style='color:{SLATE};margin-top:-0.5rem'>"
        "Upload the redacted PDF and its <b>.gocalma</b> key file to inspect or restore the original."
        "</p>",
        unsafe_allow_html=True,
    )

    col_pdf, col_key = st.columns(2)
    with col_pdf:
        deredact_pdf = st.file_uploader(
            "Redacted PDF", type=["pdf"], key="deredact_pdf",
        )
    with col_key:
        key_file = st.file_uploader(
            "Key file (.gocalma)", type=["gocalma"], key="unredact_key",
        )

    unredact_pw = st.text_input(
        "Passphrase (leave blank if none was set)",
        type="password",
        key="unredact_password",
        help="Enter the passphrase used when the key file was created, or leave blank.",
    )

    if key_file is not None and deredact_pdf is not None:
        try:
            blob = key_file.read()
            password_arg = unredact_pw if unredact_pw else None
            key, ciphertext = load_key_file(blob, password=password_arg)
            mapping = decrypt_mapping(ciphertext, key)

            pdf_data = deredact_pdf.read()
            mode = detect_redaction_mode(pdf_data, mapping)

            if mode == "reversible":
                st.success(
                    f"**Reversible redaction detected** — {len(mapping)} redaction(s) "
                    "can be restored to the original."
                )

                if st.button("Restore Original", type="primary", use_container_width=True):
                    with st.spinner("Removing redaction annotations..."):
                        restored_bytes, removed = unredact_pdf(pdf_data, mapping)
                    st.session_state["deredact_restored"] = restored_bytes
                    st.session_state["deredact_removed"] = removed

                restored_bytes = st.session_state.get("deredact_restored")
                if restored_bytes is not None:
                    removed = st.session_state.get("deredact_removed", 0)
                    st.success(f"Removed **{removed}** annotation(s).")

                    total = page_count(pdf_data)
                    if total > 1:
                        dr_page = st.number_input(
                            "Page", min_value=1, max_value=total, value=1, key="deredact_page",
                        ) - 1
                    else:
                        dr_page = 0

                    col_before, col_after = st.columns(2)
                    with col_before:
                        with st.container(border=True):
                            st.markdown(
                                "<div class='pdf-card-label'>Redacted</div>",
                                unsafe_allow_html=True,
                            )
                            st.image(render_page(pdf_data, dr_page), use_container_width=True)
                    with col_after:
                        with st.container(border=True):
                            st.markdown(
                                "<div class='pdf-card-label'>Restored</div>",
                                unsafe_allow_html=True,
                            )
                            st.image(render_page(restored_bytes, dr_page), use_container_width=True)

                    st.divider()
                    st.download_button(
                        "Download Restored PDF",
                        data=restored_bytes,
                        file_name="restored.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.warning(
                    "**Permanent redaction detected.** The original text was destroyed "
                    "during redaction and cannot be restored. The mapping below shows "
                    "what was redacted, for reference only."
                )
                display_mapping = {k: v for k, v in mapping.items() if not k.startswith("__")}
                st.json(display_mapping)

        except Exception as exc:
            st.error(f"Failed to decrypt: {_sanitize_error(exc)}")

else:
    # -----------------------------------------------------------------------
    # Step 1: Upload
    # -----------------------------------------------------------------------

    st.header("Upload your document")
    st.markdown(
        f"<p style='color:{SLATE};margin-top:-0.5rem'>"
        "Drop a PDF below to get started. Sensitive information will be detected automatically."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="pdf_upload", label_visibility="collapsed",
    )

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
            st.session_state.pdf_name = uploaded.name
            st.session_state.step = "extract"

    # -----------------------------------------------------------------------
    # Step 2: Extract + Detect
    # -----------------------------------------------------------------------

    if st.session_state.step == "extract" and st.session_state.pdf_bytes:
        try:
            with st.spinner("Extracting text from PDF..."):
                pages = extract_text(st.session_state.pdf_bytes)
                st.session_state.pages = pages

            ocr_pages = sum(1 for p in pages if p.is_ocr)
            total = len(pages)
            st.info(f"Extracted text from **{total}** page(s) ({ocr_pages} via OCR).")

            with st.spinner("Detecting PII with regex + multilingual BERT NER..."):
                entities = detect_pii_all_pages(pages)
            st.info(f"NER found **{len(entities)}** entities.")

            doc_type = "general"
            if is_llm_available():
                with st.spinner("Classifying document type..."):
                    doc_type = classify_document(pages)
                st.info(f"Document classified as **{doc_type}**")
                st.session_state.doc_type = doc_type

                with st.spinner(f"LLM verifying entities with {get_llm_model()}..."):
                    entities, _ = llm_verify_entities_batch(
                        entities, pages, doc_type=doc_type,
                    )

            disputed = [e for e in entities if "review" in e.analysis.lower()]
            llm_new = [e for e in entities if e.source == "LLM"]

            entities.sort(key=lambda e: (-e.score, e.page_num, e.start))

            approved_defaults = [True for _ in entities]

            st.session_state.entities = entities
            st.session_state.approved = approved_defaults

            detection_warnings = _warning_collector.flush_warnings()
            ocr_failed = sum(1 for p in pages if p.is_ocr and not p.text.strip())

            risk = build_summary([{"type": e.entity_type} for e in entities])
            st.session_state.risk_summary = risk

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
                "Please check that the NER model is installed (pip install transformers), "
                "then try uploading the file again."
            )

    # -----------------------------------------------------------------------
    # Step 3: Review (interactive live preview)
    # -----------------------------------------------------------------------

    if st.session_state.step == "review":
        st.header("Review detected PII")
        st.markdown(
            f"<p style='color:{SLATE};margin-top:-0.5rem'>"
            "Toggle findings in the table or <b>double-click any word</b> on the "
            "original to add / remove redactions."
            "</p>",
            unsafe_allow_html=True,
        )

        entities: list[PIIEntity] = st.session_state.entities
        approved: list[bool] = st.session_state.approved
        approach = st.session_state.get("deid_approach", DEFAULT_APPROACH)
        pdf_bytes = st.session_state.pdf_bytes
        total_pages = page_count(pdf_bytes)

        # Page selector
        if total_pages > 1:
            current_page = st.number_input(
                "Page", min_value=1, max_value=total_pages, value=1, key="preview_page"
            ) - 1
        else:
            current_page = 0

        # Word-level data for the interactive viewer
        pages_data = st.session_state.get("pages")
        words = extract_words(pdf_bytes, current_page, pages=pages_data)
        word_to_entity, redacted_indices = map_words_to_entities(
            pdf_bytes, current_page, words, entities, approved, pages=pages_data,
        )
        active_entities = [e for e, ok in zip(entities, approved) if ok]
        active_on_page = sum(1 for e in active_entities if e.page_num == current_page)

        image_b64 = base64.b64encode(render_page(pdf_bytes, current_page)).decode()

        # Side-by-side PDF cards
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
                    f"<div class='pdf-card-label'>Preview — {approach} · "
                    f"{active_on_page} redaction(s)</div>",
                    unsafe_allow_html=True,
                )
                preview_png = render_preview(
                    pdf_bytes, current_page, active_entities, approach, pages=pages_data,
                )
                st.image(preview_png, use_container_width=True)

        # Handle double-click from the interactive viewer
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

        # Risk summary + entity table
        render_risk_summary(st.session_state.get("risk_summary"))
        st.divider()
        approved = render_entity_table(entities, approved)
        st.session_state.approved = approved

        # Action buttons
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
                st.warning(
                    "Passphrase has leading/trailing spaces — make sure this is intentional."
                )

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

    # -----------------------------------------------------------------------
    # Step 4: Redact & Download
    # -----------------------------------------------------------------------

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

        _stem = st.session_state.get("pdf_name", "document.pdf").rsplit(".", 1)[0]
        _pdf_name = st.session_state.get("pdf_name", "document.pdf")

        # -- Build audit record --
        entities = st.session_state.entities or []
        approved = st.session_state.approved or []
        selected = [ent for ent, ok in zip(entities, approved) if ok]
        approach = st.session_state.get("deid_approach", DEFAULT_APPROACH)
        audit_record = create_audit(
            entities=selected,
            action_taken=approach,
            model_used="Davlan/bert-base-multilingual-cased-ner-hrl",
            llm_used=is_llm_available(),
            filename=_pdf_name,
        )

        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Download Redacted PDF",
            data=redacted_bytes,
            file_name=f"{_stem}_redacted.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        col2.download_button(
            "Download Key File",
            data=st.session_state.key_blob,
            file_name=f"{_stem}_redaction_key.gocalma",
            mime="application/octet-stream",
            use_container_width=True,
        )
        col3.download_button(
            "Download Audit Log (.json)",
            data=audit_to_bytes(audit_record),
            file_name=f"{_stem}_audit.json",
            mime="application/json",
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
st.markdown(
    "<div class='footer-tag'>GoCalma Redact &mdash; privacy, locally.</div>",
    unsafe_allow_html=True,
)
