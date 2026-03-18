"""Apply de-identification to a PDF and build the reversibility mapping."""

from __future__ import annotations

import hashlib
import uuid

import fitz  # pymupdf
from cryptography.fernet import Fernet

from gocalma.pdf_extract import PageText, WordBox
from gocalma.pii_detect import PIIEntity

APPROACHES = {
    "redact": "Remove PII completely (black box, no text)",
    "replace": "Replace with entity type label, e.g. <PERSON>",
    "mask": "Replace with masking characters (****)",
    "hash": "Replace with a SHA-256 hash of the value",
    "encrypt": "Replace with an encrypted label (reversible via key file)",
    "highlight": "Yellow highlight over PII (text stays visible)",
    "synthesize": "Replace with synthetic placeholder text",
}

DEFAULT_APPROACH = "replace"

_SYNTH_MAP = {
    "PERSON": "John Doe",
    "EMAIL_ADDRESS": "redacted@example.com",
    "PHONE_NUMBER": "+1-000-000-0000",
    "LOCATION": "123 Redacted St",
    "IBAN_CODE": "XX00 0000 0000 0000",
    "CREDIT_CARD": "0000-0000-0000-0000",
    "DATE_TIME": "01/01/2000",
    "IP_ADDRESS": "0.0.0.0",
    "US_SSN": "000-00-0000",
    "US_BANK_NUMBER": "000000000",
    "US_DRIVER_LICENSE": "X0000000",
    "US_PASSPORT": "000000000",
}


def _replacement_text(ent: PIIEntity, approach: str, label: str = "") -> str:
    """Compute the overlay/replacement string for a given approach.

    ``label`` is the unique tracking label already assigned to this entity
    (e.g. ``[PERSON_a3f2c1]``).  It is embedded by the *encrypt* approach so
    the overlay is visually distinctive without requiring per-token crypto —
    actual reversal is provided by the encrypted key file.
    """
    if approach == "redact":
        return ""
    if approach == "replace":
        return f"<{ent.entity_type}>"
    if approach == "mask":
        return "*" * len(ent.text)
    if approach == "hash":
        h = hashlib.sha256(ent.text.encode()).hexdigest()[:12]
        return f"[#{h}]"
    if approach == "encrypt":
        # The actual value is stored encrypted in the key file.
        # Embed the label so redacted areas can be identified.
        short = label[1:9] if label.startswith("[") else label[:8]
        return f"[enc:{short}]"
    if approach == "highlight":
        return ent.text
    if approach == "synthesize":
        return _SYNTH_MAP.get(ent.entity_type, "<REDACTED>")
    return f"<{ent.entity_type}>"


# ---------------------------------------------------------------------------
# OCR bounding-box lookup
# ---------------------------------------------------------------------------

def _find_rects_for_entity(
    page: fitz.Page,
    ent: PIIEntity,
    page_text: PageText | None,
) -> list[fitz.Rect]:
    """Return bounding boxes for *ent* on *page*.

    For native-text PDFs, uses ``page.search_for`` (fast, accurate).
    For OCR pages, uses the character-offset index baked into
    ``PageText.word_boxes`` so coordinates are always correct even when
    the PDF has no text layer.
    """
    # Try the text layer first (works for all non-image PDFs).
    rects = page.search_for(ent.text)
    if rects:
        return rects

    # Fall back to OCR word-box lookup.
    if page_text is None or not page_text.word_boxes:
        return []

    # Find every word box that overlaps with the entity's character span.
    matching = sorted(
        [wb for wb in page_text.word_boxes
         if wb.char_start < ent.end and wb.char_end > ent.start],
        key=lambda w: (w.y0, w.x0),
    )
    if not matching:
        return []

    # Group word boxes by line (similar y-coordinate) so multi-line entities
    # produce per-line rectangles instead of one giant bounding box.
    lines: list[list[WordBox]] = []
    for wb in matching:
        line_height = wb.y1 - wb.y0
        if lines and abs(wb.y0 - lines[-1][0].y0) < line_height * 0.5:
            lines[-1].append(wb)
        else:
            lines.append([wb])

    return [
        fitz.Rect(
            min(w.x0 for w in line),
            min(w.y0 for w in line),
            max(w.x1 for w in line),
            max(w.y1 for w in line),
        )
        for line in lines
    ]


def _page_text_for(pages: list[PageText] | None, page_num: int) -> PageText | None:
    """Look up the PageText object for *page_num* from a pages list."""
    if not pages:
        return None
    for pt in pages:
        if pt.page_num == page_num:
            return pt
    return None


# ---------------------------------------------------------------------------
# Page rendering (for UI previews)
# ---------------------------------------------------------------------------

def render_page(pdf_bytes: bytes, page_num: int, dpi: int = 150) -> bytes:
    """Render an original PDF page as PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[page_num].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_preview(
    pdf_bytes: bytes,
    page_num: int,
    entities: list[PIIEntity],
    approach: str = DEFAULT_APPROACH,
    dpi: int = 150,
    pages: list[PageText] | None = None,
) -> bytes:
    """Render a page with de-identification overlays (non-destructive preview).

    ``pages`` should be the full list of PageText objects produced by
    ``extract_text``.  When supplied, OCR pages use word-box coordinates
    for accurate overlay placement instead of a text-layer search.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num]
        page_text = _page_text_for(pages, page_num)
        page_ents = [e for e in entities if e.page_num == page_num]

        for ent in page_ents:
            rects = _find_rects_for_entity(page, ent, page_text)
            if not rects:
                continue

            overlay = _replacement_text(ent, approach)

            for rect in rects:
                if approach == "highlight":
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.85, 0))
                    annot.update()
                else:
                    shape = page.new_shape()
                    shape.draw_rect(rect)
                    shape.finish(fill=(0, 0, 0), color=(0, 0, 0))
                    shape.commit()

                    if overlay:
                        fontsize = min(7, rect.height * 0.7)
                        if fontsize >= 3:
                            page.insert_textbox(
                                fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1),
                                overlay,
                                fontsize=fontsize,
                                color=(1, 1, 1),
                                align=fitz.TEXT_ALIGN_CENTER,
                            )

        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()


def extract_words(
    pdf_bytes: bytes,
    page_num: int,
    pages: list[PageText] | None = None,
) -> list[dict]:
    """Extract every word on a page with percentage-based bounding boxes.

    For OCR pages, words and coordinates come from ``PageText.word_boxes``
    rather than the PDF text layer (which is absent for image-only PDFs).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num]
        pw, ph = page.rect.width, page.rect.height

        page_text = _page_text_for(pages, page_num)
        if page_text and page_text.word_boxes:
            # OCR path: use bounding boxes captured during OCR.
            words = []
            for i, wb in enumerate(page_text.word_boxes):
                words.append({
                    "index": i,
                    "text":  wb.text,
                    "x_pct": wb.x0 / pw * 100,
                    "y_pct": wb.y0 / ph * 100,
                    "w_pct": (wb.x1 - wb.x0) / pw * 100,
                    "h_pct": (wb.y1 - wb.y0) / ph * 100,
                    "x0": wb.x0, "y0": wb.y0,
                    "x1": wb.x1, "y1": wb.y1,
                })
            return words

        # Native-text path.
        raw = page.get_text("words")
    finally:
        doc.close()

    words = []
    for i, (x0, y0, x1, y1, text, *_rest) in enumerate(raw):
        text = text.strip()
        if not text:
            continue
        words.append({
            "index": i,
            "text": text,
            "x_pct": x0 / pw * 100,
            "y_pct": y0 / ph * 100,
            "w_pct": (x1 - x0) / pw * 100,
            "h_pct": (y1 - y0) / ph * 100,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        })
    return words


def map_words_to_entities(
    pdf_bytes: bytes,
    page_num: int,
    words: list[dict],
    entities: list[PIIEntity],
    approved: list[bool],
    pages: list[PageText] | None = None,
) -> tuple[dict[int, int], list[int]]:
    """Map word indices to entity list indices and compute redacted word indices.

    Returns:
        word_to_entity: {word_index: entity_list_index}
        redacted_indices: word indices covered by active (approved) entities
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num]
        page_text = _page_text_for(pages, page_num)
        entity_rects: list[tuple[int, fitz.Rect]] = []
        for i, ent in enumerate(entities):
            if ent.page_num != page_num:
                continue
            for rect in _find_rects_for_entity(page, ent, page_text):
                entity_rects.append((i, rect))
    finally:
        doc.close()

    word_to_entity: dict[int, int] = {}
    redacted: list[int] = []

    for w in words:
        cx = (w["x0"] + w["x1"]) / 2
        cy = (w["y0"] + w["y1"]) / 2
        for ent_idx, rect in entity_rects:
            if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
                word_to_entity[w["index"]] = ent_idx
                if approved[ent_idx]:
                    redacted.append(w["index"])
                break

    return word_to_entity, redacted


# ---------------------------------------------------------------------------
# Final redaction (destructive, produces the downloadable PDF)
# ---------------------------------------------------------------------------

def redact_pdf(
    pdf_bytes: bytes,
    entities: list[PIIEntity],
    approach: str = DEFAULT_APPROACH,
    pages: list[PageText] | None = None,
    flatten: bool = True,
) -> tuple[bytes, dict[str, str]]:
    """Apply de-identification to the supplied PII entities in the PDF.

    ``pages`` should be the full list of PageText objects produced by
    ``extract_text``.  Required for correct redaction of OCR/scanned pages.

    Args:
        flatten: If True (default), permanently removes the text layer via
                 ``apply_redactions()``.  The original text can only be
                 recovered using the encrypted ``.gocalma`` key file.
                 If False, adds opaque annotation overlays that visually
                 cover PII but preserve the underlying text.  Removing the
                 annotations in any PDF editor reveals the original.

    Returns:
        output_pdf_bytes: the processed PDF.
        mapping: dict mapping replacement labels to original text (for the
                 encrypted key file).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        mapping: dict[str, str] = {}

        is_highlight    = approach == "highlight"
        is_redact_blank = approach == "redact"

        entities_by_page: dict[int, list[PIIEntity]] = {}
        for ent in entities:
            entities_by_page.setdefault(ent.page_num, []).append(ent)

        for page_num, page_entities in entities_by_page.items():
            page      = doc[page_num]
            page_text = _page_text_for(pages, page_num)

            for ent in page_entities:
                rects = _find_rects_for_entity(page, ent, page_text)
                if not rects:
                    continue

                label   = f"[{ent.entity_type}_{uuid.uuid4().hex[:6]}]"
                mapping[label] = ent.text
                overlay = _replacement_text(ent, approach, label)

                for rect in rects:
                    if flatten:
                        # ── Permanent (flattened) mode ──────────────
                        if is_highlight:
                            annot = page.add_highlight_annot(rect)
                            annot.set_colors(stroke=(1, 0.9, 0))
                            annot.update()
                        else:
                            page.add_redact_annot(
                                rect,
                                text="" if is_redact_blank else overlay,
                                fontsize=7,
                                fill=(0, 0, 0),
                                text_color=(1, 1, 1),
                            )
                    else:
                        # ── Reversible (annotation) mode ───────────
                        if is_highlight:
                            annot = page.add_highlight_annot(rect)
                            annot.set_colors(stroke=(1, 0.9, 0))
                            annot.set_info(content=f"{ent.entity_type}: {ent.text}")
                            annot.update()
                        elif is_redact_blank or not overlay:
                            # Opaque black rectangle — hides text visually
                            annot = page.add_rect_annot(rect)
                            annot.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
                            annot.set_opacity(1.0)
                            annot.set_info(content=f"{ent.entity_type}")
                            annot.update()
                        else:
                            # Text overlay — shows replacement on black bg
                            annot = page.add_freetext_annot(
                                rect,
                                overlay,
                                fontsize=min(7, rect.height * 0.7),
                                text_color=(1, 1, 1),
                                fill_color=(0, 0, 0),
                            )
                            annot.set_border(width=0)
                            annot.set_opacity(1.0)
                            annot.set_info(content=f"{ent.entity_type}: {ent.text}")
                            annot.update()

            # Only flatten pages that used redact annotations.
            if flatten and not is_highlight:
                # PDF_REDACT_IMAGE_NONE (0): keep images, draw black fill on top.
                # PDF_REDACT_IMAGE_REMOVE (1) would delete the entire page image
                # on scanned/OCR PDFs, leaving a blank white page.
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        output_bytes = doc.tobytes(deflate=True)
    finally:
        doc.close()

    return output_bytes, mapping
