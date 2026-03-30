"""Review table and risk summary rendering for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

from gocalma.pii_detect import PIIEntity
from ui.styles import ENTITY_COLORS, SLATE, WHITE


def render_risk_summary(risk: dict | None) -> None:
    """Render the risk severity banner above the entity table."""
    if not risk:
        return

    sev = risk["severity"]
    if sev == "critical":
        _bg, _prefix, _fg = "#e74c3c", "CRITICAL — ", "#fff"
    elif sev == "moderate":
        _bg, _prefix, _fg = "#f39c12", "", "#000"
    else:
        _bg, _prefix, _fg = "#27ae60", "", "#fff"

    pills_html = ""
    for ct in risk.get("critical_types_found", []):
        pills_html += (
            f"<span style='display:inline-block;background:rgba(255,255,255,0.25);"
            f"color:{_fg};padding:2px 8px;border-radius:12px;font-size:0.7rem;"
            f"font-weight:600;margin-right:4px;margin-top:6px'>{ct}</span>"
        )

    st.markdown(
        f"<div style='background:{_bg};color:{_fg};padding:1rem 1.2rem;"
        f"border-radius:10px;margin-bottom:1rem'>"
        f"<strong>{_prefix}</strong>{risk['sentence']}"
        f"<br>{pills_html}</div>",
        unsafe_allow_html=True,
    )


def render_entity_table(
    entities: list[PIIEntity],
    approved: list[bool],
) -> list[bool]:
    """Render the interactive entity review table.

    Returns the (possibly updated) approved list.
    """
    if not entities:
        st.warning("No PII detected in this document.")
        return approved

    active_count = sum(approved)
    with st.expander(
        f"**Detected entities ({active_count} / {len(entities)} active)**",
        expanded=True,
    ):
        col_header = st.columns([0.3, 1.5, 1.8, 0.6, 1.8, 0.6, 0.8])
        col_header[0].markdown(
            "<div class='review-header'>✓</div>", unsafe_allow_html=True
        )
        col_header[1].markdown(
            "<div class='review-header'>Entity</div>", unsafe_allow_html=True
        )
        col_header[2].markdown(
            "<div class='review-header'>Detected text</div>", unsafe_allow_html=True
        )
        col_header[3].markdown(
            "<div class='review-header'>Source</div>", unsafe_allow_html=True
        )
        col_header[4].markdown(
            "<div class='review-header'>Analysis</div>", unsafe_allow_html=True
        )
        col_header[5].markdown(
            "<div class='review-header'>Page</div>", unsafe_allow_html=True
        )
        col_header[6].markdown(
            "<div class='review-header'>Confidence</div>", unsafe_allow_html=True
        )

        for i, ent in enumerate(entities):
            cols = st.columns([0.3, 1.5, 1.8, 0.6, 1.8, 0.6, 0.8])
            approved[i] = cols[0].checkbox(
                "", value=approved[i], key=f"chk_{i}", label_visibility="collapsed"
            )
            bg, fg = ENTITY_COLORS.get(ent.entity_type, (SLATE, WHITE))
            cols[1].markdown(
                f"<span class='entity-badge' style='background:{bg};color:{fg}'>"
                f"{ent.entity_type}</span>",
                unsafe_allow_html=True,
            )
            cols[2].code(ent.text)

            src = getattr(ent, "source", "NER")
            src_bg = "#3d5a80" if src == "NER" else "#8b5a6b"
            cols[3].markdown(
                f"<span class='entity-badge' style='background:{src_bg};color:#fff'>"
                f"{src}</span>",
                unsafe_allow_html=True,
            )

            # Analysis column: show LLM verdict as badge
            analysis_text = getattr(ent, "analysis", "")
            if "review" in analysis_text.lower():
                cols[4].markdown(
                    "<span style='font-size:0.75rem;background:#f39c12;color:#000;"
                    "padding:2px 8px;border-radius:8px'>⚠ review</span>",
                    unsafe_allow_html=True,
                )
            elif "verified" in analysis_text.lower():
                cols[4].markdown(
                    "<span style='font-size:0.75rem;color:#27ae60'>✓ verified</span>",
                    unsafe_allow_html=True,
                )
            else:
                cols[4].markdown(
                    f"<span style='font-size:0.75rem;color:{SLATE}'>"
                    f"{analysis_text}</span>",
                    unsafe_allow_html=True,
                )

            cols[5].caption(f"Page {ent.page_num + 1}")

            # Confidence column: colored bar with label
            if ent.score >= 0.90:
                conf_color, conf_label = "#27ae60", "High"
            elif ent.score >= 0.70:
                conf_color, conf_label = "#f39c12", "Medium"
            else:
                conf_color, conf_label = "#95a5a6", "Low"
            cols[6].markdown(
                f"<div style='background:#eee;border-radius:6px;overflow:hidden;"
                f"height:18px;margin-top:4px'>"
                f"<div style='background:{conf_color};width:{ent.score*100:.0f}%;"
                f"height:100%;border-radius:6px;display:flex;align-items:center;"
                f"justify-content:center;font-size:0.65rem;color:#fff;"
                f"font-weight:600'>{conf_label}</div></div>",
                unsafe_allow_html=True,
            )

    return approved
