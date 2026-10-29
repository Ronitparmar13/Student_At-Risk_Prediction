"""Batch Predictions page — CSV upload, schema validation, batch inference, export.

Entry point: ``render_batch_predict(mode)``

Model inputs are separated from metadata columns before inference.
Row alignment is preserved: output row N corresponds to input row N.
Failed rows appear in the output with NaN predictions; they are never
silently converted into valid predictions.
Results are stored in ``st.session_state["batch_result_df"]`` so the
Risk Review page can consume them without re-running.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from config.theme import TOKENS, LIGHT
from services.data_validation import validate_batch_csv
from services.export_service import build_predictions_csv
from services.model_service import (
    BatchPredictionResult,
    ModelInferenceError,
    ModelLoadError,
    get_streamlit_cached_pipeline,
    predict_batch,
    validate_batch_columns,
)

_SESSION_RESULT_KEY = "batch_result_df"
_SESSION_PROBA_KEY  = "batch_has_proba"


def render_batch_predict(mode: str = "light") -> None:
    """Render the Batch Predictions page.

    Parameters
    ----------
    mode: Active theme.
    """
    t = TOKENS.get(mode, LIGHT)

    st.markdown(
        f'<h2 style="color:{t["text_primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:800;margin-bottom:4px;">Batch Predictions</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{t["text_secondary"]};margin-bottom:20px;">'
        "Upload a CSV containing all 26 model input features to run predictions "
        "on multiple students at once.</p>",
        unsafe_allow_html=True,
    )

    # ── Load model (cached) ────────────────────────────────────────────────
    try:
        _pipeline, schema = get_streamlit_cached_pipeline()
    except ModelLoadError as exc:
        st.error(
            f"**Model could not be loaded.** {exc}\n\n"
            "Ensure `best_student_risk_model.pkl` is present and "
            "`scikit-learn==1.6.1` is installed."
        )
        return

    # ── Required columns info ──────────────────────────────────────────────
    with st.expander("Required columns (26 model input features)", expanded=False):
        feat_df = pd.DataFrame({
            "Feature": schema.feature_names,
            "Type": [
                "Categorical" if f in schema.categorical_features else "Numeric"
                for f in schema.feature_names
            ],
            "Known categories / range": [
                str(schema.category_values[f]) if f in schema.categorical_features
                else _range_hint(f)
                for f in schema.feature_names
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.caption(
            "Columns Student_ID, Risk_Score, Gender and At_Risk are not model inputs "
            "but may be included in the upload — they will pass through to the output."
        )

    # ── File uploader ──────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload batch CSV",
        type=["csv"],
        key="batch_uploader",
        help="Must contain all 26 model input features.",
    )

    if uploaded is None:
        st.info("Upload a CSV to begin batch predictions.")
        _show_previous_results(mode, t, schema)
        return

    # ── Read and validate ──────────────────────────────────────────────────
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    val = validate_batch_csv(raw_df, schema.feature_names)

    for warn in val.warnings:
        st.info(f"ℹ️ {warn}")

    if not val.is_valid:
        for err in val.errors:
            st.error(err)
        return

    st.success(
        f"✅ File accepted — {len(raw_df):,} rows, "
        f"{raw_df.shape[1]} columns.",
        icon=None,
    )

    # ── Run predictions ────────────────────────────────────────────────────
    if st.button("▶ Run predictions", type="primary", key="batch_run"):
        with st.spinner(f"Running predictions on {len(raw_df):,} rows…"):
            try:
                result: BatchPredictionResult = predict_batch(raw_df)
            except ModelInferenceError as exc:
                st.error(f"**Batch inference failed:** {exc}")
                return

        # Store in session state for Risk Review page
        st.session_state[_SESSION_RESULT_KEY] = result.output_df
        st.session_state[_SESSION_PROBA_KEY] = "Estimated_Probability" in result.output_df.columns

        n_ok    = len(raw_df) - len(result.failed_indices)
        n_fail  = len(result.failed_indices)
        n_risk  = int((result.output_df["Predicted_At_Risk"] == 1).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Input rows",       f"{len(raw_df):,}")
        c2.metric("Predicted",        f"{n_ok:,}")
        c3.metric("Flagged at-risk",  f"{n_risk:,}")

        if result.failed_indices:
            st.warning(
                f"⚠️ {n_fail} row(s) could not be predicted (shown as NaN below). "
                "These rows are included in the output CSV with blank prediction columns."
            )
            with st.expander(f"Failed rows ({n_fail})", expanded=False):
                for reason in result.failed_reasons:
                    st.caption(reason)

        st.success("Predictions complete. Download the results below.")

    # ── Show results if available ──────────────────────────────────────────
    _show_previous_results(mode, t, schema)


def _show_previous_results(mode: str, t: dict, schema) -> None:
    """Display and allow download of batch results from session state."""
    result_df: pd.DataFrame | None = st.session_state.get(_SESSION_RESULT_KEY)
    has_proba: bool = st.session_state.get(_SESSION_PROBA_KEY, False)

    if result_df is None:
        return

    st.divider()
    st.markdown(
        f'<div style="color:{t["text_primary"]};font-weight:700;margin-bottom:8px;">'
        f"Prediction results — {len(result_df):,} rows</div>",
        unsafe_allow_html=True,
    )

    # Filter controls
    filter_cols = st.columns([1, 1, 2])
    with filter_cols[0]:
        class_filter = st.selectbox(
            "Show class",
            ["All", "At risk (1)", "Not at risk (0)", "Failed (NaN)"],
            key="batch_class_filter",
        )
    with filter_cols[1]:
        if has_proba:
            prob_min = st.slider(
                "Min probability",
                0.0, 1.0, 0.0, 0.01,
                key="batch_prob_filter",
            )
        else:
            prob_min = 0.0

    # Apply filters
    display = result_df.copy()
    if class_filter == "At risk (1)":
        display = display[display["Predicted_At_Risk"] == 1]
    elif class_filter == "Not at risk (0)":
        display = display[display["Predicted_At_Risk"] == 0]
    elif class_filter == "Failed (NaN)":
        display = display[display["Predicted_At_Risk"].isna()]

    if has_proba and prob_min > 0.0:
        prob_col = display["Estimated_Probability"]
        display = display[prob_col.notna() & (prob_col >= prob_min)]

    st.caption(f"Showing {len(display):,} of {len(result_df):,} rows")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Download
    predictions_list = list(result_df["Predicted_At_Risk"])
    proba_list = (
        list(result_df["Estimated_Probability"]) if has_proba else None
    )
    # Reconstruct input_df without the appended prediction columns
    input_cols = [
        c for c in result_df.columns
        if c not in ("Predicted_At_Risk", "Estimated_Probability")
    ]
    input_df_only = result_df[input_cols]

    csv_bytes = build_predictions_csv(input_df_only, predictions_list, proba_list)
    st.download_button(
        "⬇️ Download predictions CSV",
        csv_bytes,
        file_name="batch_predictions.csv",
        mime="text/csv",
        key="batch_download",
    )

    if has_proba:
        st.caption(
            "ℹ️ **Estimated_Probability** is an uncalibrated RandomForest output — "
            "the fraction of decision trees that voted for class 1 (at-risk). "
            "It is not guaranteed confidence and has not been calibrated against "
            "independent outcome data."
        )


def _range_hint(col: str) -> str:
    """Return a human-readable range hint for numeric features."""
    from views.student_predict import _get_range  # avoid circular at module level
    lo, hi = _get_range(col)
    return f"{lo} – {hi}"
