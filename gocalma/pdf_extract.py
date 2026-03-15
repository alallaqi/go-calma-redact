"""PDF text extraction with OCR fallback for scanned documents.

OCR engine priority (auto-selected, first available wins):
  1. Surya  — transformer-based, 90+ languages, no language flag needed
  2. Tesseract — classic, requires system binary + language packs
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field

import fitz  # pymupdf

# Rendering DPI used for OCR.  200 DPI gives ~2 000×2 800 px for an A4 page
# (≈ 6 MB uncompressed) — sufficient accuracy while avoiding the ≈ 26 MB/page
# footprint of 300 DPI.  Callers can override via the ``ocr_dpi`` parameter.
_DEFAULT_OCR_DPI = 200


@dataclass
class WordBox:
    """A single OCR word with its bounding box in PDF coordinate space (points).

    ``char_start`` / ``char_end`` are character offsets into the full page
    text string stored in ``PageText.text``, so NER entity spans map directly
    onto these boxes without any secondary text search.
    """
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    char_start: int
    char_end: int


@dataclass
class PageText:
    page_num: int
    text: str
    is_ocr: bool
    # Populated only for OCR pages; None for native-text pages.
    word_boxes: list[WordBox] | None = field(default=None)


# ---------------------------------------------------------------------------
# Surya model cache (loaded once, reused across pages)
# ---------------------------------------------------------------------------

_surya_cache: dict = {}
_surya_lock  = threading.Lock()


def _surya_available() -> bool:
    """Return True if surya-ocr is importable."""
    try:
        import surya.ocr  # noqa: F401
        return True
    except Exception:
        return False


def _load_surya() -> dict | None:
    """Lazily load and cache Surya detection + recognition models."""
    if "ready" in _surya_cache:
        return _surya_cache if _surya_cache.get("ready") else None

    with _surya_lock:
        if "ready" in _surya_cache:
            return _surya_cache if _surya_cache.get("ready") else None
        try:
            from surya.model.detection.model import load_model as load_det
            from surya.model.detection.model import load_processor as load_det_proc
            from surya.model.recognition.model import load_model as load_rec
            from surya.model.recognition.processor import load_processor as load_rec_proc

            _surya_cache["det_model"]      = load_det()
            _surya_cache["det_processor"]  = load_det_proc()
            _surya_cache["rec_model"]      = load_rec()
            _surya_cache["rec_processor"]  = load_rec_proc()
            _surya_cache["ready"]          = True
        except Exception:
            _surya_cache["ready"] = False

    return _surya_cache if _surya_cache.get("ready") else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(
    pdf_bytes: bytes,
    ocr_if_empty: bool = True,
    ocr_dpi: int = _DEFAULT_OCR_DPI,
    ocr_lang: str = "deu+fra+ita+eng",
) -> list[PageText]:
    """Extract text from every page.

    Falls back to OCR (Surya preferred, Tesseract as fallback) when a page
    has no selectable text.  The document handle is always closed, even if
    an exception occurs mid-way.

    Args:
        pdf_bytes:    Raw PDF bytes.
        ocr_if_empty: Run OCR on pages that have no embedded text.
        ocr_dpi:      Rendering resolution for OCR (default 200 DPI).
        ocr_lang:     Tesseract language string — ignored when Surya is used
                      (Surya auto-detects language).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[PageText] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if text:
                pages.append(PageText(page_num=page_num, text=text, is_ocr=False))
            elif ocr_if_empty:
                ocr_text, word_boxes = _ocr_page(page, dpi=ocr_dpi, lang=ocr_lang)
                pages.append(PageText(
                    page_num=page_num,
                    text=ocr_text,
                    is_ocr=True,
                    word_boxes=word_boxes,
                ))
            else:
                pages.append(PageText(page_num=page_num, text="", is_ocr=False))
    finally:
        doc.close()

    return pages


def _ocr_page(
    page: fitz.Page,
    dpi: int = _DEFAULT_OCR_DPI,
    lang: str = "deu+fra+ita+eng",
) -> tuple[str, list[WordBox]]:
    """Render *page* and run OCR with word-level bounding boxes.

    Tries Surya first (transformer-based, language-agnostic), falls back to
    Tesseract.  Returns ``("", [])`` if neither engine is available.

    The returned text and word_boxes character offsets are always consistent:
    NER entity ``start``/``end`` spans map directly to word boxes via
    ``char_start``/``char_end``.
    """
    if _surya_available():
        result = _ocr_page_surya(page, dpi=dpi)
        if result[0]:   # non-empty text means Surya succeeded
            return result

    return _ocr_page_tesseract(page, dpi=dpi, lang=lang)


# ---------------------------------------------------------------------------
# Surya backend
# ---------------------------------------------------------------------------

def _ocr_page_surya(page: fitz.Page, dpi: int = _DEFAULT_OCR_DPI) -> tuple[str, list[WordBox]]:
    """OCR via Surya — transformer-based, supports 90+ languages automatically."""
    try:
        from surya.ocr import run_ocr
        from PIL import Image
    except ImportError:
        return "", []

    models = _load_surya()
    if models is None:
        return "", []

    try:
        pix      = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        scale_x  = page.rect.width  / pix.width
        scale_y  = page.rect.height / pix.height
        del pix

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        results = run_ocr(
            [img],
            [["de", "fr", "it", "en"]],
            models["det_model"],
            models["det_processor"],
            models["rec_model"],
            models["rec_processor"],
        )
        if not results:
            return "", []

        text_parts: list[str] = []
        word_boxes: list[WordBox] = []
        current_pos = 0

        for line in results[0].text_lines:
            line_text = (line.text or "").strip()
            if not line_text:
                continue

            # Compute line bbox from polygon [[x,y], ...]
            poly = line.polygon
            lx0  = min(p[0] for p in poly)
            ly0  = min(p[1] for p in poly)
            lx1  = max(p[0] for p in poly)
            ly1  = max(p[1] for p in poly)

            if text_parts:
                text_parts.append("\n")
                current_pos += 1

            words = line_text.split()
            if not words:
                continue

            # Distribute word x-positions proportionally by character length.
            line_px_width = lx1 - lx0
            total_chars   = sum(len(w) for w in words)

            x_cursor = lx0
            for j, word in enumerate(words):
                if j > 0:
                    text_parts.append(" ")
                    current_pos += 1

                word_ratio = len(word) / total_chars if total_chars else 1 / len(words)
                wx0 = x_cursor
                wx1 = x_cursor + line_px_width * word_ratio
                x_cursor = wx1

                char_start = current_pos
                text_parts.append(word)
                current_pos += len(word)

                word_boxes.append(WordBox(
                    text=word,
                    x0=wx0 * scale_x,
                    y0=ly0 * scale_y,
                    x1=wx1 * scale_x,
                    y1=ly1 * scale_y,
                    char_start=char_start,
                    char_end=current_pos,
                ))

        return "".join(text_parts).strip(), word_boxes

    except Exception:
        return "", []


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------

def _ocr_page_tesseract(
    page: fitz.Page,
    dpi: int = _DEFAULT_OCR_DPI,
    lang: str = "deu+fra+ita+eng",
) -> tuple[str, list[WordBox]]:
    """OCR via Tesseract — requires system binary + language packs.

    Returns ``("", [])`` if pytesseract / Pillow are absent or OCR fails.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", []

    try:
        pix      = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        scale_x  = page.rect.width  / pix.width
        scale_y  = page.rect.height / pix.height
        del pix

        img  = Image.open(io.BytesIO(png_bytes))
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

        text_parts: list[str] = []
        word_boxes: list[WordBox] = []
        current_pos = 0
        prev_block: int | None = None
        prev_line:  int | None = None

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            if not word or int(data["conf"][i]) < 0:
                continue

            block = data["block_num"][i]
            line  = data["line_num"][i]

            if text_parts:
                if block != prev_block:
                    sep = "\n\n"
                elif line != prev_line:
                    sep = "\n"
                else:
                    sep = " "
                text_parts.append(sep)
                current_pos += len(sep)

            char_start = current_pos
            text_parts.append(word)
            current_pos += len(word)

            word_boxes.append(WordBox(
                text=word,
                x0=data["left"][i] * scale_x,
                y0=data["top"][i] * scale_y,
                x1=(data["left"][i] + data["width"][i]) * scale_x,
                y1=(data["top"][i] + data["height"][i]) * scale_y,
                char_start=char_start,
                char_end=current_pos,
            ))

            prev_block = block
            prev_line  = line

        return "".join(text_parts).strip(), word_boxes

    except Exception:
        return "", []


def get_text_positions(pdf_bytes: bytes, search_text: str, page_num: int) -> list[fitz.Rect]:
    """Find all bounding-box positions of *search_text* on a given page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num]
        return page.search_for(search_text)
    finally:
        doc.close()
