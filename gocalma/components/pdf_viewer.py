"""Bi-directional Streamlit component: interactive PDF page viewer."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"
_component = components.declare_component("pdf_viewer", path=str(_FRONTEND_DIR))


def pdf_viewer(
    image_base64: str,
    words: list[dict],
    redacted_indices: list[int],
    approach: str,
    key: str | None = None,
):
    """Render an interactive PDF page with hoverable / double-clickable word boxes.

    Returns None when idle, or a dict ``{word_index, text, ts}`` after a double-click.
    """
    return _component(
        image_base64=image_base64,
        words=words,
        redacted_indices=redacted_indices,
        approach=approach,
        key=key,
        default=None,
    )
