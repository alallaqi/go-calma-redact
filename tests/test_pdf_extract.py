"""Tests for gocalma.pdf_extract — text extraction, OCR fallback, page limits."""

from __future__ import annotations

import pytest
import fitz

from gocalma.pdf_extract import (
    PageText,
    WordBox,
    extract_text,
    MAX_PAGES,
    _surya_available,
)


# ---------------------------------------------------------------------------
# Helpers — create minimal PDFs in memory
# ---------------------------------------------------------------------------

def _make_pdf(pages: list[str]) -> bytes:
    """Create a minimal PDF with the given page texts."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        if text:
            page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _make_empty_image_pdf(num_pages: int = 1) -> bytes:
    """Create a PDF with blank image pages (no text layer)."""
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page(width=595, height=842)
        # Draw a filled rect so the page isn't completely empty visually,
        # but there's no text layer → triggers OCR path.
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(100, 100, 200, 200))
        shape.finish(fill=(0.5, 0.5, 0.5))
        shape.commit()
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

class TestWordBox:
    def test_construction(self):
        wb = WordBox(text="hello", x0=10, y0=20, x1=50, y1=35,
                     char_start=0, char_end=5)
        assert wb.text == "hello"
        assert wb.x0 == 10
        assert wb.char_end == 5

    def test_width_and_height(self):
        wb = WordBox(text="x", x0=10, y0=20, x1=50, y1=40,
                     char_start=0, char_end=1)
        assert wb.x1 - wb.x0 == 40
        assert wb.y1 - wb.y0 == 20


class TestPageText:
    def test_native_text_page(self):
        pt = PageText(page_num=0, text="Hello world", is_ocr=False)
        assert pt.page_num == 0
        assert pt.text == "Hello world"
        assert pt.is_ocr is False
        assert pt.word_boxes is None

    def test_ocr_page_with_word_boxes(self):
        boxes = [WordBox("Hi", 0, 0, 10, 10, 0, 2)]
        pt = PageText(page_num=1, text="Hi", is_ocr=True, word_boxes=boxes)
        assert pt.is_ocr is True
        assert len(pt.word_boxes) == 1


# ---------------------------------------------------------------------------
# extract_text — native text PDFs
# ---------------------------------------------------------------------------

class TestExtractTextNative:
    def test_single_page(self):
        pdf = _make_pdf(["Hello World"])
        pages = extract_text(pdf, ocr_if_empty=False)
        assert len(pages) == 1
        assert "Hello" in pages[0].text
        assert pages[0].is_ocr is False

    def test_multi_page(self):
        pdf = _make_pdf(["Page one", "Page two", "Page three"])
        pages = extract_text(pdf, ocr_if_empty=False)
        assert len(pages) == 3
        assert pages[0].page_num == 0
        assert pages[2].page_num == 2

    def test_empty_page_without_ocr(self):
        pdf = _make_pdf(["Text", ""])
        pages = extract_text(pdf, ocr_if_empty=False)
        assert len(pages) == 2
        assert pages[1].text == ""
        assert pages[1].is_ocr is False

    def test_page_text_content_preserved(self):
        text = "Max Mustermann, Musterstrasse 1, 8003 Zürich"
        pdf = _make_pdf([text])
        pages = extract_text(pdf, ocr_if_empty=False)
        # PyMuPDF may strip/reformat slightly, but core content is there
        assert "Mustermann" in pages[0].text
        assert "8003" in pages[0].text


# ---------------------------------------------------------------------------
# extract_text — MAX_PAGES limit
# ---------------------------------------------------------------------------

class TestMaxPages:
    def test_constant_is_reasonable(self):
        assert MAX_PAGES >= 100
        assert MAX_PAGES <= 10000

    def test_rejects_pdf_over_limit(self):
        # Create a small PDF with text pages to exceed a low limit
        pdf = _make_pdf(["text"] * 5)
        with pytest.raises(ValueError, match="limit"):
            extract_text(pdf, max_pages=3, ocr_if_empty=False)

    def test_accepts_pdf_at_limit(self):
        pdf = _make_pdf(["text"] * 3)
        pages = extract_text(pdf, max_pages=3, ocr_if_empty=False)
        assert len(pages) == 3

    def test_accepts_pdf_under_limit(self):
        pdf = _make_pdf(["text"] * 2)
        pages = extract_text(pdf, max_pages=5, ocr_if_empty=False)
        assert len(pages) == 2


# ---------------------------------------------------------------------------
# OCR availability check
# ---------------------------------------------------------------------------

class TestOcrAvailability:
    def test_surya_available_returns_bool(self):
        assert isinstance(_surya_available(), bool)

    def test_empty_page_triggers_ocr_flag(self):
        """An image-only page with ocr_if_empty=True should be marked is_ocr."""
        pdf = _make_empty_image_pdf(1)
        pages = extract_text(pdf, ocr_if_empty=True)
        assert len(pages) == 1
        assert pages[0].is_ocr is True
