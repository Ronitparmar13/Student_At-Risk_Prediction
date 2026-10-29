"""Prediction indicator component — accessible risk badge and probability display.

``render_prediction(predicted_class, probability, mode)`` is the only public
function.  It renders:

  1. A pill-shaped badge: "⚠️ AT RISK" or "✅ NOT AT RISK", using colour AND
     text so the result is never communicated by colour alone.
  2. A probability gauge (``st.progress``) and labelled percentage, shown only
     when ``probability`` is not None.
  3. A fixed calibration disclaimer and an educator-review reminder.

Probability labelling rules (per approved specification):
  - Always labelled "Estimated model probability (uncalibrated)".
  - Never described as "confidence", "guaranteed", or "model accuracy".
  - The calibration limitation is disclosed every time probability is shown.
  - If ``probability`` is None, the class result is shown without any
    percentage, and a note explains that probability is unavailable.
"""
from __future__ import annotations

import streamlit as st

from config.theme import TOKENS, LIGHT


# ── Constants ──────────────────────────────────────────────────────────────

_CALIBRATION_NOTE = (
    "ℹ️ **Calibration note:** This is an uncalibrated RandomForest probability "
    "estimate. It reflects the model's internal vote fraction and has not been "
    "calibrated against independent outcome data. It is not a guaranteed "
    "confidence score and should not be treated as equivalent to model accuracy."
)

_REVIEW_REMINDER = (
    "🔔 **Educator review required:** This output is a decision-support signal "
    "only. A qualified educator must review the student's full context before "
    "taking any action. Do not use this result as the sole basis for "
    "consequential decisions about any student."
)


# ── Public API ─────────────────────────────────────────────────────────────

def render_prediction(
    predicted_class: int,
    probability: float | None,
    mode: str = "light",
) -> None:
    """Render the risk badge, optional probability gauge, and disclaimers.

    Parameters
    ----------
    predicted_class:
        Integer 0 (not at risk) or 1 (at risk).
    probability:
        Estimated uncalibrated probability for class 1 (at-risk), in [0, 1].
        Pass ``None`` to suppress the gauge and show the class result only.
    mode:
        Active theme mode — ``"light"`` or ``"dark"``.
    """
    t = TOKENS.get(mode, LIGHT)

    # ── 1. Risk badge ──────────────────────────────────────────────────────
    is_at_risk = (predicted_class == 1)

    badge_color      = t["at_risk"]  if is_at_risk else t["not_risk"]
    badge_text_color = t["at_risk_text"] if is_at_risk else t["not_risk_text"]
    badge_icon       = "⚠️" if is_at_risk else "✅"
    badge_label      = "AT RISK" if is_at_risk else "NOT AT RISK"

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: {badge_color};
                color: {badge_text_color};
                padding: 10px 24px;
                border-radius: 50px;
                font-family: 'Manrope', sans-serif;
                font-size: 1.05rem;
                font-weight: 800;
                letter-spacing: 0.04em;
            ">
                <span style="font-size:1.2rem;">{badge_icon}</span>
                {badge_label}
            </span>
            <div style="
                margin-top: 8px;
                color: {t['text_secondary']};
                font-size: 0.82rem;
            ">
                Predicted class: <strong style="color:{t['text_primary']};">{predicted_class}</strong>
                ({badge_label.lower()})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Probability gauge ───────────────────────────────────────────────
    if probability is not None:
        _validate_probability(probability)

        pct = int(round(probability * 100))

        st.markdown(
            f"""
            <div style="margin-bottom: 4px; color: {t['text_secondary']}; font-size: 0.83rem;">
                Estimated model probability (uncalibrated)
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Streamlit progress bar (0–100)
        st.progress(pct)

        st.markdown(
            f"""
            <div style="
                font-family: 'Manrope', sans-serif;
                font-size: 1.45rem;
                font-weight: 800;
                color: {t['text_primary']};
                margin-bottom: 4px;
            ">{pct}%</div>
            <div style="color: {t['text_secondary']}; font-size: 0.80rem; margin-bottom: 12px;">
                estimated probability the model assigns to class 1 (at-risk)
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(_CALIBRATION_NOTE)

    else:
        # Probability not available
        st.markdown(
            f"""
            <div style="
                color: {t['text_secondary']};
                font-size: 0.84rem;
                margin-bottom: 12px;
                padding: 10px 14px;
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 8px;
            ">
                ℹ️ Estimated probability not available for this prediction.
                The predicted class is shown above.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 3. Educator review reminder (always shown) ─────────────────────────
    st.warning(_REVIEW_REMINDER)


# ── Internal helpers ───────────────────────────────────────────────────────

def _validate_probability(probability: float) -> None:
    """Raise ValueError if the probability is outside [0, 1]."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError(
            f"Probability must be in [0.0, 1.0], got {probability}. "
            "Do not pass unchecked model output directly to the indicator."
        )


# ── HTML-only rendering (for tests and non-Streamlit contexts) ─────────────

def prediction_badge_html(
    predicted_class: int,
    probability: float | None,
    mode: str = "light",
) -> str:
    """Return the badge HTML string without calling Streamlit.

    Used by tests to verify HTML structure and token injection without
    requiring a running Streamlit session.
    """
    t = TOKENS.get(mode, LIGHT)
    is_at_risk = (predicted_class == 1)
    badge_color      = t["at_risk"]  if is_at_risk else t["not_risk"]
    badge_text_color = t["at_risk_text"] if is_at_risk else t["not_risk_text"]
    badge_icon       = "⚠️" if is_at_risk else "✅"
    badge_label      = "AT RISK" if is_at_risk else "NOT AT RISK"

    prob_html = ""
    if probability is not None:
        pct = int(round(probability * 100))
        prob_html = (
            f'<div class="at-prob">{pct}% estimated model probability (uncalibrated)</div>'
        )

    return (
        f'<span class="at-badge" style="background:{badge_color};color:{badge_text_color};">'
        f"{badge_icon} {badge_label}</span>"
        f"{prob_html}"
    )
