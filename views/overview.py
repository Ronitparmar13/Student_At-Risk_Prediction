"""Overview page — dashboard landing with live KPIs, charts, and dataset preview.

Entry point: ``render_overview(df, meta, mode)``

All displayed figures are derived from the caller-supplied DataFrame.
No placeholder metrics, fabricated counts, or example data are shown.
When no dataset is uploaded, an upload-guidance empty state is rendered.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import bar_chart, donut_chart
from components.kpi_cards import KPICard, render_kpi_row
from config.theme import TOKENS, LIGHT
from services.analytics import (
    compute_kpis,
    group_risk_rates,
    leakage_check,
)
from services.data_loader import CleaningMetadata
from services.data_validation import validate_dataset
from services.export_service import build_cleaned_csv

_GROUP_COLS = ["Stream", "Higher_Education_Interest", "Gender"]
_SCORE_COLS = [
    "Attendance_Percentage",
    "Assignment_Completion_Percentage",
    "Previous_Semester_GPA",
    "Previous_Assessment_Score",
]


def render_overview(
    df: pd.DataFrame | None,
    meta: CleaningMetadata | None = None,
    mode: str = "light",
) -> None:
    """Render the Overview page.

    Parameters
    ----------
    df:    Cleaned DataFrame from ``data_loader``, or ``None`` when no file
           has been uploaded yet.
    meta:  ``CleaningMetadata`` from ``data_loader``.  Used for the cleaning
           summary.  May be ``None``.
    mode:  Active theme — ``"light"`` or ``"dark"``.
    """
    t = TOKENS.get(mode, LIGHT)

    st.markdown(
        f'<h2 style="color:{t["text_primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:800;margin-bottom:4px;">Dashboard Overview</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{t["text_secondary"]};margin-bottom:20px;">'
        "A snapshot of the uploaded dataset — distribution, quality, and at-risk patterns.</p>",
        unsafe_allow_html=True,
    )

    # ── Empty state ────────────────────────────────────────────────────────
    if df is None or len(df) == 0:
        _render_empty_state(t)
        return

    # ── Validation advisories ──────────────────────────────────────────────
    val = validate_dataset(df)
    for err in val.errors:
        st.error(err)
    for warn in val.warnings:
        st.warning(warn)

    # ── KPI row ────────────────────────────────────────────────────────────
    kpis = compute_kpis(df)
    cards = _build_kpi_cards(kpis)
    render_kpi_row(cards, mode=mode)
    st.write("")

    # ── Row 1: donut + score bar chart ─────────────────────────────────────
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(
            f'<div style="color:{t["text_primary"]};font-weight:700;'
            f'margin-bottom:8px;">At-Risk Distribution</div>',
            unsafe_allow_html=True,
        )
        if kpis.has_target and kpis.at_risk_count is not None:
            fig = donut_chart(
                labels=["Not at risk", "At risk"],
                values=[kpis.not_at_risk_count, kpis.at_risk_count],
                mode=mode,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a dataset with an `At_Risk` column to see the distribution.")

    with right:
        st.markdown(
            f'<div style="color:{t["text_primary"]};font-weight:700;'
            f'margin-bottom:8px;">Key Academic Metrics (mean by At-Risk status)</div>',
            unsafe_allow_html=True,
        )
        _render_score_comparison(df, mode, t)

    st.divider()

    # ── Row 2: group risk rates ────────────────────────────────────────────
    available_groups = [c for c in _GROUP_COLS if c in df.columns]
    if available_groups and "At_Risk" in df.columns:
        st.markdown(
            f'<div style="color:{t["text_primary"]};font-weight:700;'
            f'margin-bottom:8px;">At-Risk Rate by Group</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Group comparisons are descriptive only. "
            "They do not imply causal relationships or fairness assessments."
        )
        grp_col = st.selectbox(
            "Group by",
            available_groups,
            key="overview_group_select",
            label_visibility="collapsed",
        )
        grp_result = group_risk_rates(df, grp_col)
        if not grp_result.summary.empty:
            grp_pct = grp_result.summary.copy()
            grp_pct["At_Risk_Rate_%"] = (grp_pct["At_Risk_Rate"] * 100).round(1)
            fig = bar_chart(
                grp_pct,
                x_col=grp_col,
                y_col="At_Risk_Rate_%",
                mode=mode,
                y_label="At-Risk Rate (%)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                grp_pct[[grp_col, "Record_Count", "At_Risk_Count", "At_Risk_Rate_%"]]
                .rename(columns={"At_Risk_Rate_%": "At-Risk Rate (%)"}),
                use_container_width=True,
                hide_index=True,
            )
        st.divider()

    # ── Leakage advisory ──────────────────────────────────────────────────
    leak = leakage_check(df)
    if leak is not None:
        if leak == 1.0:
            st.warning(
                "⚠️ **Data advisory:** `Risk_Score >= 50` matches `At_Risk` for "
                "every row. This suggests the target may be rule-generated from "
                "`Risk_Score`. Dataset provenance is unverified. "
                "`Risk_Score` is excluded from model inputs."
            )
        elif leak >= 0.95:
            st.info(
                f"ℹ️ `Risk_Score >= 50` matches `At_Risk` for {leak:.1%} of rows. "
                "This is an advisory, not confirmed provenance."
            )

    # ── Cleaning summary ───────────────────────────────────────────────────
    if meta is not None:
        with st.expander("Cleaning summary", expanded=False):
            st.markdown(
                f'<p style="color:{t["text_secondary"]};font-size:0.88rem;">'
                f"Raw rows: <strong>{meta.raw_row_count:,}</strong> &nbsp;|&nbsp; "
                f"Duplicates removed: <strong>{meta.duplicate_rows_removed:,}</strong> &nbsp;|&nbsp; "
                f"Rows dropped (target): <strong>{meta.missing_target_rows_dropped + meta.invalid_target_rows_dropped:,}</strong> &nbsp;|&nbsp; "
                f"Clean rows: <strong>{meta.clean_row_count:,}</strong>"
                f"</p>",
                unsafe_allow_html=True,
            )
            if meta.out_of_range_counts:
                oor = pd.DataFrame(
                    [{"Column": k, "Out-of-range values → NaN": v}
                     for k, v in meta.out_of_range_counts.items()]
                )
                st.dataframe(oor, use_container_width=True, hide_index=True)

    # ── Dataset preview ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{t["text_primary"]};font-weight:700;'
        f'margin-bottom:8px;">Dataset Preview (first 10 rows)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    csv_bytes = build_cleaned_csv(df)
    st.download_button(
        "⬇️ Download cleaned CSV",
        csv_bytes,
        file_name="student_at_risk_cleaned.csv",
        mime="text/csv",
        key="overview_download",
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_kpi_cards(kpis) -> list[KPICard]:
    cards = [
        KPICard(
            label="Total Records",
            value=f"{kpis.record_count:,}",
            note="After cleaning",
            icon="📊",
        ),
        KPICard(
            label="At-Risk Students",
            value=f"{kpis.at_risk_count:,}" if kpis.at_risk_count is not None else None,
            note=(
                f"{kpis.at_risk_pct:.1%} of records"
                if kpis.at_risk_pct is not None else "At_Risk column required"
            ),
            icon="⚠️",
        ),
        KPICard(
            label="Data Completeness",
            value=f"{kpis.completeness:.1%}",
            note=f"{kpis.missing_cells:,} missing cells",
            icon="✅",
        ),
        KPICard(
            label="Columns",
            value=str(kpis.column_count),
            note=(
                f"{kpis.unique_students:,} unique Student IDs"
                if kpis.unique_students is not None else "No Student_ID column"
            ),
            icon="📋",
        ),
    ]
    return cards


def _render_score_comparison(df: pd.DataFrame, mode: str, t: dict) -> None:
    """Bar chart: mean score of key academic features split by At_Risk."""
    available = [c for c in _SCORE_COLS if c in df.columns]
    if not available or "At_Risk" not in df.columns:
        st.info("No key academic columns found for comparison.")
        return

    rows = []
    for col in available:
        for cls, label in [(0, "Not at risk"), (1, "At risk")]:
            subset = df[df["At_Risk"] == cls][col].dropna()
            if len(subset):
                rows.append({
                    "Feature": col.replace("_", " "),
                    "Group": label,
                    "Mean": round(float(subset.mean()), 2),
                })
    if not rows:
        st.info("Insufficient data for score comparison.")
        return

    import pandas as _pd
    import plotly.express as px
    from components.charts import apply_chart_theme
    compare_df = _pd.DataFrame(rows)
    fig = px.bar(
        compare_df,
        x="Feature",
        y="Mean",
        color="Group",
        barmode="group",
        color_discrete_map={
            "Not at risk": t["not_risk"],
            "At risk":     t["at_risk"],
        },
        labels={"Mean": "Mean value", "Feature": ""},
    )
    fig.update_layout(
        margin=dict(t=16, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_chart_theme(fig, mode)
    st.plotly_chart(fig, use_container_width=True)


def _render_empty_state(t: dict) -> None:
    st.markdown(
        f"""
        <div style="
            background:{t['surface']};
            border:1px solid {t['border']};
            border-radius:14px;
            padding:40px 32px;
            text-align:center;
            margin-top:16px;
        ">
            <div style="font-size:2.5rem;margin-bottom:12px;">📂</div>
            <div style="
                font-family:Manrope,sans-serif;
                font-size:1.2rem;
                font-weight:700;
                color:{t['text_primary']};
                margin-bottom:8px;
            ">No dataset uploaded yet</div>
            <div style="color:{t['text_secondary']};font-size:0.92rem;max-width:420px;margin:0 auto;">
                Upload a student CSV from the sidebar to see KPIs, charts, and group summaries.
                The file must include an <code>At_Risk</code> column with values 0 and 1.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    _step_card(c1, "1 — Upload", "Add your student CSV from the sidebar uploader.", t)
    _step_card(c2, "2 — Explore", "Review data quality, distributions, and group patterns.", t)
    _step_card(c3, "3 — Predict", "Run single or batch predictions using the saved model.", t)


def _step_card(col, title: str, body: str, t: dict) -> None:
    col.markdown(
        f"""<div style="
            background:{t['surface']};border:1px solid {t['border']};
            border-radius:12px;padding:18px 16px;text-align:center;
        ">
            <div style="font-weight:700;color:{t['primary']};margin-bottom:6px;">{title}</div>
            <div style="color:{t['text_secondary']};font-size:0.84rem;">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )
