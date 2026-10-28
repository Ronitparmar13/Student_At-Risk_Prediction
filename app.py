"""Student At-Risk Analytics Dashboard — application entry point.

This file handles only: page configuration, session-state initialisation,
theme injection, and page routing. All business logic lives in services/,
all UI components in components/, and all page layouts in views/.
"""
from __future__ import annotations
import streamlit as st
from config.theme import apply_theme

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Success | At-Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session-state initialisation ───────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

# ── Theme injection ────────────────────────────────────────────────────────
st.markdown(apply_theme(st.session_state["theme"]), unsafe_allow_html=True)

# ── Hero banner with theme toggle ──────────────────────────────────────────
_mode = st.session_state["theme"]
_toggle_label = "🌙 Dark" if _mode == "light" else "☀️ Light"
_toggle_next = "dark" if _mode == "light" else "light"

# Use columns: left = title area, right = toggle button
_hero_left, _hero_right = st.columns([5, 1])

with _hero_left:
    st.markdown(
        """
        <div class="at-hero-left" style="padding: 20px 0 16px 4px;">
            <div class="eyebrow">TEAM 4 · STUDENT SUCCESS ANALYTICS</div>
            <h1>Student At-Risk Prediction</h1>
            <p>Explore student performance data, audit data quality, and review risk predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with _hero_right:
    st.write("")  # spacing
    st.write("")
    if st.button(_toggle_label, key="theme_toggle", use_container_width=True):
        st.session_state["theme"] = _toggle_next
        st.rerun()

st.divider()

# ── Milestone 0 placeholder ────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="at-card" style="max-width:600px; margin: 40px auto; text-align:center;">
        <div class="label">STATUS</div>
        <div class="value">Milestone 0 Complete</div>
        <div class="note">Scaffold ready · Theme shell active · Model verified</div>
        <div style="margin-top:12px; color:#64748B; font-size:0.9rem;">
            Active theme: <strong>{_mode}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Full navigation and pages will be wired in Milestone 3. "
    "Prediction, data explorer, and all six pages are coming."
)
