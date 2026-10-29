"""Predict a Student page — single-record prediction using the saved model.

Entry point: ``render_student_predict(df, mode)``

The form is built entirely from the live FeatureSchema introspected from the
fitted pipeline.  No feature names, order, or categorical values are hard-coded
here.  The page works without an uploaded dataset if the model loads cleanly.

Student_ID, Risk_Score, Gender, and At_Risk are never presented as input fields.
Probabilities are always labelled "estimated model probability (uncalibrated)".
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.prediction_indicator import render_prediction
from config.theme import TOKENS, LIGHT
from services.model_service import (
    ModelInferenceError,
    ModelLoadError,
    get_streamlit_cached_pipeline,
    predict_single,
)

# Numeric field range constraints derived from the notebook cleaning rules.
# Used to set sensible min/max on number_input widgets.
_RANGE_HINTS: dict[str, tuple[float, float]] = {}

_PERCENT_SUFFIX = "Percentage"
_SCORE_SUFFIX   = "_Score"
_SOFT_SKILLS    = frozenset({
    "Analytical_Thinking", "Creativity_Level",
    "Attention_to_Detail", "Communication_Skills",
})


def _get_range(col: str) -> tuple[float, float]:
    if col == "Previous_Semester_GPA":
        return 0.0, 10.0
    if col == "Age":
        return 14.0, 100.0
    if col in _SOFT_SKILLS:
        return 1.0, 10.0
    if _PERCENT_SUFFIX in col or col.endswith(_SCORE_SUFFIX):
        return 0.0, 100.0
    return 0.0, 100.0   # Safe fallback


def _get_step(col: str) -> float:
    if col == "Previous_Semester_GPA":
        return 0.1
    if col in _SOFT_SKILLS:
        return 1.0
    if col == "Age":
        return 1.0
    return 0.5


def render_student_predict(
    df: pd.DataFrame | None = None,
    mode: str = "light",
) -> None:
    """Render the Predict a Student page.

    Parameters
    ----------
    df:    Cleaned DataFrame (optional — used only for default value hints).
           Prediction works without a dataset if the model loads successfully.
    mode:  Active theme.
    """
    t = TOKENS.get(mode, LIGHT)

    st.markdown(
        f'<h2 style="color:{t["text_primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:800;margin-bottom:4px;">Predict a Student</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{t["text_secondary"]};margin-bottom:20px;">'
        "Enter a student's feature values to receive a risk estimate from the saved model. "
        "A dataset upload is not required.</p>",
        unsafe_allow_html=True,
    )

    # ── Load model (cached) ────────────────────────────────────────────────
    try:
        pipeline, schema = get_streamlit_cached_pipeline()
    except ModelLoadError as exc:
        st.error(
            f"**Model could not be loaded.** {exc}\n\n"
            "Ensure `best_student_risk_model.pkl` is in the project root and "
            "`scikit-learn==1.6.1` is installed."
        )
        return

    st.success(
        f"✅ Model loaded — {len(schema.feature_names)} input features, "
        f"{len(schema.categorical_features)} categorical.",
        icon=None,
    )

    # ── Build form ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{t["text_secondary"]};font-size:0.84rem;margin-bottom:12px;">'
        "Fields are populated with the midpoint of each feature's valid range. "
        "Adjust values to match the student's actual data.</div>",
        unsafe_allow_html=True,
    )

    with st.form("predict_single_form"):
        input_values: dict = {}
        cols_iter = iter(st.columns(2))

        for feat in schema.feature_names:
            col = next(cols_iter, None)
            if col is None:
                cols_iter = iter(st.columns(2))
                col = next(cols_iter)

            label = feat.replace("_", " ")

            if feat in schema.categorical_features:
                known_cats = schema.category_values.get(feat, [])
                input_values[feat] = col.selectbox(
                    label,
                    options=known_cats,
                    key=f"pred_{feat}",
                )
            else:
                lo, hi = _get_range(feat)
                default = (lo + hi) / 2.0
                # If a dataset is loaded, use its median as the default
                if df is not None and feat in df.columns:
                    med = df[feat].median()
                    if pd.notna(med):
                        default = float(med)
                input_values[feat] = col.number_input(
                    label,
                    min_value=lo,
                    max_value=hi,
                    value=round(default, 2),
                    step=_get_step(feat),
                    key=f"pred_{feat}",
                )

        submitted = st.form_submit_button(
            "🎯 Estimate risk",
            type="primary",
            use_container_width=True,
        )

    # ── Run prediction ─────────────────────────────────────────────────────
    if submitted:
        try:
            predicted_class, probability = predict_single(input_values)
        except ModelInferenceError as exc:
            st.error(f"**Prediction failed:** {exc}")
            return

        st.divider()
        left, right = st.columns([1, 1.4])

        with left:
            render_prediction(predicted_class, probability, mode=mode)

        with right:
            st.markdown(
                f'<div style="color:{t["text_secondary"]};font-size:0.85rem;">'
                "<strong>What these fields mean:</strong><br><br>"
                "<b>Predicted class 0</b> — the model's vote is 'not at risk' for this input.<br>"
                "<b>Predicted class 1</b> — the model's vote is 'at risk' for this input.<br><br>"
                "The probability shown is the <em>uncalibrated</em> fraction of RandomForest "
                "decision trees that voted for class 1. It is not model accuracy and is not "
                "a guaranteed confidence score.<br><br>"
                "This output is a <strong>decision-support signal only</strong>. "
                "A qualified educator must review the student's full context."
                "</div>",
                unsafe_allow_html=True,
            )
