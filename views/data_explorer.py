"""Data Explorer page — searchable records, schema, quality, distributions, heatmap.

Entry point: ``render_data_explorer(df, meta, mode)``

Risk_Score and At_Risk are excluded from the feature correlation heatmap
to avoid presenting leakage-related correlations as informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import bar_chart, box_chart, heatmap_chart, histogram_chart
from config.theme import TOKENS, LIGHT
from services.analytics import (
    correlation_matrix,
    data_quality_report,
    group_risk_rates,
)
from services.data_loader import CleaningMetadata
from services.export_service import build_cleaned_csv, build_quality_report_csv

_EXCLUDE_DIST = {"At_Risk", "Risk_Score", "Student_ID"}
_GROUP_COLS   = ["Stream", "Higher_Education_Interest", "Gender"]


def render_data_explorer(
    df: pd.DataFrame | None,
    meta: CleaningMetadata | None = None,
    mode: str = "light",
) -> None:
    """Render the Data Explorer page.

    Parameters
    ----------
    df:    Cleaned DataFrame, or ``None``.
    meta:  ``CleaningMetadata`` from ``data_loader``.
    mode:  Active theme.
    """
    t = TOKENS.get(mode, LIGHT)

    st.markdown(
        f'<h2 style="color:{t["text_primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:800;margin-bottom:4px;">Data Explorer</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{t["text_secondary"]};margin-bottom:20px;">'
        "Inspect records, data quality, numeric distributions, and feature correlations.</p>",
        unsafe_allow_html=True,
    )

    if df is None or len(df) == 0:
        _empty_state(t)
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Records",
        "🗂 Schema",
        "🔎 Data Quality",
        "📈 Distributions",
        "🔗 Correlations",
    ])

    # ── Tab 1: Searchable records ──────────────────────────────────────────
    with tab1:
        _render_records_tab(df, mode, t)

    # ── Tab 2: Schema ──────────────────────────────────────────────────────
    with tab2:
        _render_schema_tab(df, t)

    # ── Tab 3: Data quality ────────────────────────────────────────────────
    with tab3:
        _render_quality_tab(df, meta, t)

    # ── Tab 4: Distributions ───────────────────────────────────────────────
    with tab4:
        _render_distributions_tab(df, mode, t)

    # ── Tab 5: Correlations ────────────────────────────────────────────────
    with tab5:
        _render_correlations_tab(df, mode, t)


# ── Tab renderers ──────────────────────────────────────────────────────────

def _render_records_tab(df: pd.DataFrame, mode: str, t: dict) -> None:
    st.markdown(
        f'<p style="color:{t["text_secondary"]};font-size:0.88rem;">'
        f"{len(df):,} records · {df.shape[1]} columns</p>",
        unsafe_allow_html=True,
    )

    # Text search filter across all string columns
    search = st.text_input(
        "Filter records (searches all text columns)",
        key="explorer_search",
        placeholder="e.g. Commerce, Male …",
    )
    display_df = df.copy()
    if search.strip():
        mask = display_df.apply(
            lambda col: col.astype(str).str.contains(search.strip(), case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
        st.caption(f'{len(display_df):,} rows match "{search.strip()}"')

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = build_cleaned_csv(df)
    st.download_button(
        "⬇️ Download cleaned CSV",
        csv_bytes,
        file_name="student_at_risk_cleaned.csv",
        mime="text/csv",
        key="explorer_dl_csv",
    )


def _render_schema_tab(df: pd.DataFrame, t: dict) -> None:
    schema = pd.DataFrame({
        "Column":         df.columns,
        "Data Type":      [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [int(df[c].notna().sum()) for c in df.columns],
        "Missing":        [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values":  [int(df[c].nunique(dropna=False)) for c in df.columns],
        "Sample":         [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                           for c in df.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)


def _render_quality_tab(
    df: pd.DataFrame,
    meta: CleaningMetadata | None,
    t: dict,
) -> None:
    report = data_quality_report(df)
    st.dataframe(report, use_container_width=True, hide_index=True)

    q_bytes = build_quality_report_csv(report)
    st.download_button(
        "⬇️ Download quality report",
        q_bytes,
        file_name="student_at_risk_data_quality_report.csv",
        mime="text/csv",
        key="explorer_dl_quality",
    )

    if meta and meta.out_of_range_counts:
        st.markdown(
            f'<div style="color:{t["text_secondary"]};font-size:0.84rem;margin-top:12px;">'
            "Out-of-range values were set to NaN during cleaning:</div>",
            unsafe_allow_html=True,
        )
        oor = pd.DataFrame(
            [{"Column": k, "Values set to NaN": v}
             for k, v in meta.out_of_range_counts.items()]
        )
        st.dataframe(oor, use_container_width=True, hide_index=True)

    if meta:
        st.caption(
            f"Cleaning summary — raw: {meta.raw_row_count:,} rows · "
            f"duplicates removed: {meta.duplicate_rows_removed:,} · "
            f"dropped for target issues: "
            f"{meta.missing_target_rows_dropped + meta.invalid_target_rows_dropped:,} · "
            f"clean: {meta.clean_row_count:,}"
        )


def _render_distributions_tab(df: pd.DataFrame, mode: str, t: dict) -> None:
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in _EXCLUDE_DIST
    ]
    cat_cols = [c for c in _GROUP_COLS if c in df.columns]

    if not numeric_cols:
        st.info("No numeric columns available for distribution charts.")
        return

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            f'<div style="font-weight:600;color:{t["text_primary"]};margin-bottom:6px;">'
            "Numeric distribution</div>",
            unsafe_allow_html=True,
        )
        sel_num = st.selectbox(
            "Select feature",
            numeric_cols,
            key="dist_num_select",
            label_visibility="collapsed",
        )
        fig_hist = histogram_chart(df, sel_num, mode=mode)
        st.plotly_chart(fig_hist, use_container_width=True)

        if "At_Risk" in df.columns:
            fig_box = box_chart(
                df, "At_Risk", sel_num, mode=mode,
                x_labels={0: "Not at risk", 1: "At risk"},
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with right:
        if cat_cols and "At_Risk" in df.columns:
            st.markdown(
                f'<div style="font-weight:600;color:{t["text_primary"]};margin-bottom:6px;">'
                "Group at-risk rate</div>",
                unsafe_allow_html=True,
            )
            sel_cat = st.selectbox(
                "Group by",
                cat_cols,
                key="dist_cat_select",
                label_visibility="collapsed",
            )
            grp = group_risk_rates(df, sel_cat)
            if not grp.summary.empty:
                grp_pct = grp.summary.copy()
                grp_pct["At_Risk_Rate_%"] = (grp_pct["At_Risk_Rate"] * 100).round(1)
                fig_bar = bar_chart(
                    grp_pct,
                    x_col=sel_cat,
                    y_col="At_Risk_Rate_%",
                    mode=mode,
                    y_label="At-Risk Rate (%)",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption(
                    "Group differences are descriptive. "
                    "They do not imply causal explanations."
                )
        else:
            st.info("No categorical group columns available.")


def _render_correlations_tab(df: pd.DataFrame, mode: str, t: dict) -> None:
    st.markdown(
        f'<p style="color:{t["text_secondary"]};font-size:0.84rem;margin-bottom:12px;">'
        "Pearson correlation between numeric feature columns. "
        "<code>At_Risk</code>, <code>Risk_Score</code>, and <code>Student_ID</code> "
        "are excluded to avoid leakage-related artefacts.</p>",
        unsafe_allow_html=True,
    )
    corr = correlation_matrix(df)
    fig = heatmap_chart(corr, mode=mode)
    if corr.empty:
        st.info(
            "Not enough numeric feature columns for a correlation matrix "
            "(need at least 2 after excluding At_Risk, Risk_Score, and Student_ID)."
        )
    else:
        st.plotly_chart(fig, use_container_width=True)


def _empty_state(t: dict) -> None:
    st.markdown(
        f"""
        <div style="
            background:{t['surface']};border:1px solid {t['border']};
            border-radius:14px;padding:40px 32px;text-align:center;margin-top:16px;
        ">
            <div style="font-size:2.5rem;margin-bottom:12px;">🔍</div>
            <div style="font-family:Manrope,sans-serif;font-size:1.1rem;font-weight:700;
                        color:{t['text_primary']};margin-bottom:8px;">
                No dataset to explore
            </div>
            <div style="color:{t['text_secondary']};font-size:0.9rem;max-width:400px;margin:0 auto;">
                Upload a student CSV from the sidebar to explore records, data quality,
                distributions, and feature correlations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
