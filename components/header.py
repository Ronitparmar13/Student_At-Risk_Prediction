"""Header component — purple-gradient hero banner with top-right theme toggle.

``render_header(mode)`` is the only public function.  It:
  - renders the full-width gradient hero using the active LIGHT/DARK tokens
  - places the app title and description in the left column
  - places a ☀️ Light / 🌙 Dark toggle button in the right column
  - on toggle click sets ``st.session_state["theme"]`` and calls ``st.rerun()``

The toggle persists across page navigations because it writes to session state.
"""
from __future__ import annotations

import streamlit as st

from config.theme import TOKENS, LIGHT


# ── Public API ─────────────────────────────────────────────────────────────

def render_header(mode: str = "light") -> None:
    """Render the hero banner and theme toggle.

    Parameters
    ----------
    mode:
        Active theme mode — ``"light"`` or ``"dark"``.
        Reads/writes ``st.session_state["theme"]`` to persist the toggle.
    """
    t = TOKENS.get(mode, LIGHT)

    # Gradient uses header→primary→accent from the active token set.
    # The hero is full-width with rounded bottom corners.
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(120deg, {t['header']} 0%, {t['primary']} 55%, {t['accent']} 100%);
            padding: 22px 32px 20px 32px;
            border-radius: 0 0 18px 18px;
            margin-bottom: 0;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:16px;">
                <div>
                    <div style="
                        text-transform: uppercase;
                        letter-spacing: 0.14em;
                        color: #BDB4F0;
                        font-weight: 700;
                        font-size: 0.70rem;
                        margin-bottom: 6px;
                        font-family: 'DM Sans', sans-serif;
                    ">TEAM 4 · STUDENT SUCCESS ANALYTICS</div>
                    <h1 style="
                        font-family: 'Manrope', sans-serif;
                        color: #F4F6FC;
                        font-size: 1.85rem;
                        font-weight: 800;
                        margin: 0 0 6px 0;
                        line-height: 1.2;
                    ">Student At-Risk Prediction</h1>
                    <p style="
                        color: #D0C8F8;
                        margin: 0;
                        font-size: 0.95rem;
                        font-family: 'DM Sans', sans-serif;
                    ">Explore student performance data, audit quality, and review risk predictions.</p>
                </div>
                <div id="at-theme-toggle-placeholder"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # The toggle button must be a real Streamlit widget so it is keyboard-
    # accessible and focusable. We float it visually to the top-right via a
    # CSS override that targets only this specific button key.
    # Using CSS to visually position it doesn't break Streamlit's DOM order.
    _next_mode = "dark" if mode == "light" else "light"
    _label = "🌙 Dark" if mode == "light" else "☀️ Light"

    # Inject CSS to style only the toggle button (scoped by its key)
    st.markdown(
        f"""
        <style>
        div[data-testid="stButton"][key="at_theme_toggle"] > button,
        [data-testid="baseButton-secondary"][aria-label="{_label}"] {{
            position: fixed;
            top: 22px;
            right: 24px;
            z-index: 9999;
            background: rgba(255,255,255,0.13) !important;
            border: 1px solid rgba(255,255,255,0.30) !important;
            border-radius: 24px !important;
            color: #F4F6FC !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            padding: 5px 16px !important;
            min-height: 0 !important;
            line-height: 1.4 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render a small column pair so the button appears near the top right.
    # (Fixed-position CSS above handles the visual overlay on the hero.)
    _gap, _btn_col = st.columns([20, 1])
    with _btn_col:
        if st.button(_label, key="at_theme_toggle", use_container_width=False):
            st.session_state["theme"] = _next_mode
            st.rerun()
