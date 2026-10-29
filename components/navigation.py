"""Horizontal navigation component — exactly six approved pages.

``render_nav(mode)`` renders a themed horizontal tab bar and returns the name
of the active page.  The active page is persisted in
``st.session_state["active_page"]`` so it survives theme toggles and reruns.

Approved pages (in display order):
    Overview · Data Explorer · Predict a Student ·
    Batch Predictions · Risk Review · Project Notes

Model Lab is not present and must never be added here.
"""
from __future__ import annotations

import streamlit as st

from config.theme import TOKENS, LIGHT

# ── Constants ──────────────────────────────────────────────────────────────

PAGES: list[str] = [
    "Overview",
    "Data Explorer",
    "Predict a Student",
    "Batch Predictions",
    "Risk Review",
    "Project Notes",
]

# Emoji icons paired with each page label
_ICONS: dict[str, str] = {
    "Overview":             "🏠",
    "Data Explorer":        "🔍",
    "Predict a Student":    "🎯",
    "Batch Predictions":    "📋",
    "Risk Review":          "⚠️",
    "Project Notes":        "📝",
}


# ── Public API ─────────────────────────────────────────────────────────────

def render_nav(mode: str = "light") -> str:
    """Render the horizontal navigation bar and return the active page name.

    Parameters
    ----------
    mode:
        Active theme mode — ``"light"`` or ``"dark"``.

    Returns
    -------
    str
        The name of the currently selected page.
        Also stored in ``st.session_state["active_page"]``.
    """
    t = TOKENS.get(mode, LIGHT)

    # Ensure session state has a valid active page
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGES[0]
    current = st.session_state["active_page"]
    if current not in PAGES:
        current = PAGES[0]
        st.session_state["active_page"] = current

    # ── Navigation bar container CSS ──────────────────────────────────────
    st.markdown(
        f"""
        <style>
        /* Nav bar wrapper */
        .at-nav-bar {{
            display: flex;
            gap: 2px;
            background: {t['surface']};
            border-bottom: 2px solid {t['border']};
            padding: 0 8px;
            overflow-x: auto;
            white-space: nowrap;
            margin-bottom: 20px;
        }}
        /* Individual tab buttons */
        .at-nav-bar .stButton > button {{
            background: transparent !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            border-radius: 0 !important;
            color: {t['text_secondary']} !important;
            font-size: 0.86rem !important;
            font-weight: 600 !important;
            padding: 10px 14px 8px 14px !important;
            margin-bottom: -2px !important;
            white-space: nowrap;
            transition: color 0.15s, border-bottom-color 0.15s;
        }}
        .at-nav-bar .stButton > button:hover {{
            color: {t['primary']} !important;
            border-bottom-color: {t['border']} !important;
            background: {t['bg']} !important;
        }}
        /* Active tab highlight — injected via a separate class below */
        .at-nav-active > .stButton > button {{
            color: {t['primary']} !important;
            border-bottom: 3px solid {t['primary']} !important;
            background: {t['bg']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Render tabs as Streamlit buttons inside a flex wrapper ────────────
    # Using st.columns so buttons sit side-by-side. A leading HTML div marks
    # the start of the nav region; an empty markdown closes it.
    st.markdown('<div class="at-nav-bar">', unsafe_allow_html=True)
    cols = st.columns(len(PAGES))
    for col, page in zip(cols, PAGES):
        icon = _ICONS.get(page, "")
        label = f"{icon} {page}"
        # Wrap in the active class when this is the current page
        if page == current:
            col.markdown('<div class="at-nav-active">', unsafe_allow_html=True)
        with col:
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state["active_page"] = page
                st.rerun()
        if page == current:
            col.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state["active_page"]


def get_active_page() -> str:
    """Return the current active page without rendering anything."""
    return st.session_state.get("active_page", PAGES[0])
