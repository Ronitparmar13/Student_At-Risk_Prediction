"""Risk Review page — read-only queue of model-flagged students.

Entry point: ``render_risk_review(mode)``

Reads batch-prediction results from ``st.session_state["batch_result_df"]``
(populated by the Batch Predictions page).  If no results exist, the user
is directed to run batch predictions first.

This page is intentionally read-only:
  - No persistent review-status controls.
  - No database writes.
  - ``Student_ID`` is hidden by default; a checkbox reveals it.

Probability filter controls are shown only when probability data exists.
Model flags are not definitive judgments; this is stated on the page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.theme import TOKENS, LIGHT
from services.export_service import build_review_export

_SESSION_RESULT_KEY = "batch_result_df"
_SESSION_PROBA_KEY  = "batch_has_proba"


def render_risk_review(mode: str = "light") -> None:
    """Render the Risk Review page.

    Parameters
    ----------
    mode: Active theme.
    """
    t = TOKENS.get(mode, LIGHT)

    st.markdown(
        f'<h2 style="color:{t["text_primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:800;margin-bottom:4px;">Risk Review</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{t["text_secondary"]};margin-bottom:4px;">'
        "A read-only review queue of students flagged by the model.</p>",
        unsafe_allow_html=True,
    )
    st.warning(
        "⚠️ **Important:** Model flags are not definitive judgments about any student. "
        "A qualified educator must review each flagged case in full context before "
        "taking any action. This tool is a decision-support aid, not a diagnosis."
    )

    # ── Check for batch results ────────────────────────────────────────────
    result_df: pd.DataFrame | None = st.session_state.get(_SESSION_RESULT_KEY)

    if result_df is None or len(result_df) == 0:
        _empty_state(t)
        return

    has_proba: bool = st.session_state.get(_SESSION_PROBA_KEY, False)
    if has_proba and "Estimated_Probability" not in result_df.columns:
        has_proba = False

    # ── Summary metrics ────────────────────────────────────────────────────
    total       = len(result_df)
    n_flagged   = int((result_df["Predicted_At_Risk"] == 1).sum())
    n_not_risk  = int((result_df["Predicted_At_Risk"] == 0).sum())
    n_failed    = int(result_df["Predicted_At_Risk"].isna().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records",     f"{total:,}")
    c2.metric("Flagged at-risk",   f"{n_flagged:,}")
    c3.metric("Not at risk",       f"{n_not_risk:,}")
    c4.metric("Failed / NaN",      f"{n_failed:,}")

    st.divider()

    # ── Filter controls ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{t["text_primary"]};font-weight:700;margin-bottom:8px;">'
        "Filter flagged records</div>",
        unsafe_allow_html=True,
    )

    filter_row = st.columns([1, 1, 1, 2])

    with filter_row[0]:
        class_filter = st.radio(
            "Show",
            ["At risk", "Not at risk", "All"],
            index=0,
            key="review_class_filter",
            horizontal=False,
        )

    with filter_row[1]:
        show_student_id = st.checkbox(
            "Show Student ID",
            value=False,
            key="review_show_id",
        )

    with filter_row[2]:
        if has_proba:
            prob_threshold = st.slider(
                "Min probability",
                0.0, 1.0, 0.5, 0.01,
                key="review_prob_threshold",
            )
        else:
            prob_threshold = 0.0
            st.caption("Probability data not available.")

    # ── Apply filters ──────────────────────────────────────────────────────
    filtered = result_df.copy()

    if class_filter == "At risk":
        filtered = filtered[filtered["Predicted_At_Risk"] == 1]
    elif class_filter == "Not at risk":
        filtered = filtered[filtered["Predicted_At_Risk"] == 0]
    # "All" keeps NaN rows too

    if has_proba and prob_threshold > 0.0:
        prob_col = filtered["Estimated_Probability"]
        filtered = filtered[prob_col.notna() & (prob_col >= prob_threshold)]

    # Hide Student_ID unless user opts in
    display = filtered.copy()
    if not show_student_id and "Student_ID" in display.columns:
        display = display.drop(columns=["Student_ID"])

    # ── Results table ──────────────────────────────────────────────────────
    st.caption(f"Showing {len(display):,} of {total:,} records")
    st.dataframe(display, use_container_width=True, hide_index=True)

    if has_proba:
        st.caption(
            "ℹ️ **Estimated_Probability** is an uncalibrated RandomForest output — "
            "the fraction of trees that voted for class 1. It is not model accuracy "
            "and has not been calibrated against independent outcome data."
        )

    # ── Download ───────────────────────────────────────────────────────────
    if len(display) > 0:
        export_bytes = build_review_export(
            filtered,
            hide_student_id=(not show_student_id),
        )
        st.download_button(
            "⬇️ Download filtered review CSV",
            export_bytes,
            file_name="risk_review_export.csv",
            mime="text/csv",
            key="review_download",
        )
    else:
        st.info("No records match the current filters.")


def _empty_state(t: dict) -> None:
    st.markdown(
        f"""
        <div style="
            background:{t['surface']};border:1px solid {t['border']};
            border-radius:14px;padding:40px 32px;text-align:center;margin-top:16px;
        ">
            <div style="font-size:2.5rem;margin-bottom:12px;">📋</div>
            <div style="font-family:Manrope,sans-serif;font-size:1.1rem;font-weight:700;
                        color:{t['text_primary']};margin-bottom:8px;">
                No batch predictions available
            </div>
            <div style="color:{t['text_secondary']};font-size:0.9rem;max-width:420px;margin:0 auto;">
                Go to <strong>Batch Predictions</strong>, upload a student CSV,
                and run predictions. The results will appear here automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
