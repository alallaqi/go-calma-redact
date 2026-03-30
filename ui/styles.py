"""GoCalma brand palette and Streamlit CSS injection."""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Brand palette (matched from gocalma.ch)
# ---------------------------------------------------------------------------
NAVY = "#2b3a4e"
NAVY_DARK = "#1e2d3d"
NAVY_LIGHT = "#3d5a80"
SLATE = "#6b7b8d"
BG_ALT = "#f5f7fa"
WHITE = "#ffffff"

ENTITY_COLORS: dict[str, tuple[str, str]] = {
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


def inject_css() -> None:
    """Inject the full GoCalma CSS theme into the current Streamlit page."""
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
    /* Inline code in sidebar markdown */
    section[data-testid="stSidebar"] .stMarkdown code {{
        background: rgba(255,255,255,0.15) !important;
        color: {WHITE} !important;
        padding: 0.1em 0.4em;
        border-radius: 4px;
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
