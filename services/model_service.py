"""Model service — trusted loading and inference for best_student_risk_model.pkl.

Only this project-owned artifact is ever loaded. The module never accepts a
caller-supplied path, never loads arbitrary pickles, and never silently swallows
inference errors.

Probabilities produced by this module are uncalibrated RandomForest outputs.
They must be presented to users as "estimated model probability (uncalibrated)"
and must NOT be described as guaranteed confidence scores or model accuracy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────

# The only pkl path this module will ever load.
_PKL_PATH = Path(__file__).resolve().parent.parent / "best_student_risk_model.pkl"

# Columns that must never reach the model regardless of what the caller passes.
_FORBIDDEN_COLS: frozenset[str] = frozenset(
    {"Student_ID", "Risk_Score", "Gender", "At_Risk"}
)


# ── Custom exceptions ──────────────────────────────────────────────────────

class ModelLoadError(RuntimeError):
    """Raised when the saved model artifact cannot be loaded."""


class ModelInferenceError(RuntimeError):
    """Raised when a prediction attempt fails for a recoverable reason.

    The message is intended to be displayed directly to the user in the UI layer.
    Never catch this silently or convert it into a fake prediction result.
    """


# ── Internal schema types ──────────────────────────────────────────────────

class FeatureSchema(NamedTuple):
    """Verified schema extracted from the fitted pipeline at load time."""
    feature_names: list[str]          # All 26 input column names, in fitted order
    numeric_features: list[str]       # 24 numeric column names
    categorical_features: list[str]   # 2 categorical column names
    category_values: dict[str, list]  # {col: [known_category, ...]} from fitted OHE
    at_risk_class_index: int          # Index of class 1 in predict_proba output


# ── Pipeline loader (cached at module level for non-Streamlit callers) ─────

_cached_pipeline = None
_cached_schema: FeatureSchema | None = None


def _load_pipeline_once():
    """Load and cache the pipeline. Thread-safe for single-process Streamlit."""
    global _cached_pipeline, _cached_schema
    if _cached_pipeline is not None:
        return _cached_pipeline, _cached_schema

    if not _PKL_PATH.exists():
        raise ModelLoadError(
            f"Model artifact not found at expected path: {_PKL_PATH}. "
            "Ensure best_student_risk_model.pkl is in the project root."
        )

    try:
        pipeline = joblib.load(_PKL_PATH)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load {_PKL_PATH.name}: {exc}\n"
            "The model was saved with scikit-learn 1.6.1. "
            "Confirm scikit-learn==1.6.1 is installed (pip show scikit-learn)."
        ) from exc

    schema = _extract_schema(pipeline)
    _cached_pipeline = pipeline
    _cached_schema = schema
    return pipeline, schema


def _extract_schema(pipeline) -> FeatureSchema:
    """Introspect the fitted pipeline to derive the verified feature schema.

    Reads feature names from the ColumnTransformer and category lists from
    the fitted OneHotEncoder. Never hardcodes feature names or categories.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    # feature_names_in_ is set on the ColumnTransformer when it is fitted on
    # a DataFrame; it contains the full ordered list of input columns.
    if not hasattr(preprocessor, "feature_names_in_"):
        raise ModelLoadError(
            "ColumnTransformer is missing feature_names_in_. "
            "The model must have been fitted on a pandas DataFrame."
        )
    feature_names: list[str] = list(preprocessor.feature_names_in_)

    # Collect categorical column names and their fitted OHE categories.
    cat_cols: list[str] = []
    cat_values: dict[str, list] = {}
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = transformer.named_steps["encoder"]
            for col, cats in zip(cols, ohe.categories_):
                cat_cols.append(col)
                cat_values[col] = [str(c) for c in cats]

    num_cols = [c for c in feature_names if c not in cat_cols]

    # Locate class 1 in the estimator's classes_ array — never assume index 1.
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "classes_"):
        raise ModelLoadError(
            "Estimator is missing classes_. "
            "The model must have been fitted before saving."
        )
    at_risk_indices = np.where(estimator.classes_ == 1)[0]
    if len(at_risk_indices) == 0:
        raise ModelLoadError(
            f"Class label 1 not found in estimator.classes_: {estimator.classes_}. "
            "Cannot determine the at-risk probability index."
        )
    at_risk_idx = int(at_risk_indices[0])

    return FeatureSchema(
        feature_names=feature_names,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        category_values=cat_values,
        at_risk_class_index=at_risk_idx,
    )


# ── Public API ─────────────────────────────────────────────────────────────

def load_saved_model():
    """Load the trusted project model. Returns (pipeline, FeatureSchema).

    In a Streamlit app use get_pipeline() / get_schema() which wrap this with
    @st.cache_resource. This function is provided for non-Streamlit callers
    (tests, scripts).

    Raises ModelLoadError if the artifact cannot be loaded.
    """
    return _load_pipeline_once()


def get_feature_schema() -> FeatureSchema:
    """Return the verified FeatureSchema derived from the fitted pipeline.

    Raises ModelLoadError if the model cannot be loaded.
    """
    _, schema = _load_pipeline_once()
    return schema


def predict_single(input_values: dict) -> tuple[int, float]:
    """Run inference on a single student record.

    Parameters
    ----------
    input_values:
        Dict mapping feature name → value for all 26 model input features.
        Must not include Student_ID, Risk_Score, Gender, or At_Risk.

    Returns
    -------
    (predicted_class, at_risk_probability)
        predicted_class: int, 0 or 1
        at_risk_probability: float in [0.0, 1.0]
            Uncalibrated RF probability for class 1 (at-risk).

    Raises
    ------
    ModelInferenceError
        If any forbidden column is present, any required feature is missing,
        or the pipeline itself raises an exception. Never returns silently on
        failure.
    """
    pipeline, schema = _load_pipeline_once()

    # Guard: reject forbidden columns
    forbidden_found = _FORBIDDEN_COLS & set(input_values.keys())
    if forbidden_found:
        raise ModelInferenceError(
            f"Prediction input must not include these columns: {sorted(forbidden_found)}. "
            "Remove them and retry."
        )

    # Guard: require all expected features
    missing = set(schema.feature_names) - set(input_values.keys())
    if missing:
        raise ModelInferenceError(
            f"Missing required input feature(s): {sorted(missing)}. "
            "Provide a value for every model feature listed in the prediction form."
        )

    try:
        input_df = pd.DataFrame([input_values], columns=schema.feature_names)
        pred_array = pipeline.predict(input_df)
        proba_array = pipeline.predict_proba(input_df)
    except Exception as exc:
        raise ModelInferenceError(
            f"Inference failed: {exc}. "
            "Check that all input values are within expected ranges and try again."
        ) from exc

    predicted_class = int(pred_array[0])
    at_risk_prob = float(proba_array[0, schema.at_risk_class_index])
    return predicted_class, at_risk_prob


class BatchPredictionResult(NamedTuple):
    """Result of a batch prediction run."""
    output_df: pd.DataFrame       # Input rows + Predicted_At_Risk + Estimated_Probability
    failed_indices: list[int]     # Row indices (0-based) where inference failed
    failed_reasons: list[str]     # Parallel list of error messages for failed rows


def predict_batch(input_df: pd.DataFrame) -> BatchPredictionResult:
    """Run inference on a batch of student records.

    Parameters
    ----------
    input_df:
        DataFrame containing at minimum all 26 model input columns.
        Additional columns (identifiers, metadata) are preserved in the output
        but never passed to the model. Forbidden columns are stripped before
        inference but carried through to the output.

    Returns
    -------
    BatchPredictionResult
        output_df has the same rows as input_df (in the same order) with two
        new columns appended: Predicted_At_Risk (int or NaN on failure) and
        Estimated_Probability (float or NaN on failure).
        failed_indices and failed_reasons describe any rows where inference
        raised an exception.

    Raises
    ------
    ModelInferenceError
        If the required model input columns are entirely absent from input_df,
        indicating the wrong file was uploaded. Per-row failures are NOT raised;
        they are collected and returned in failed_indices / failed_reasons so
        callers can display them alongside the successful rows.
    """
    pipeline, schema = _load_pipeline_once()

    # Check that all required model features are present at all
    missing_cols = set(schema.feature_names) - set(input_df.columns)
    if missing_cols:
        raise ModelInferenceError(
            f"Batch CSV is missing required model input column(s): "
            f"{sorted(missing_cols)}. "
            "Upload a file that contains all 26 model input features."
        )

    # Warn if forbidden columns are present (carry them through, don't feed to model)
    forbidden_present = list(_FORBIDDEN_COLS & set(input_df.columns))

    # Build model-input-only DataFrame in fitted feature order
    model_input = input_df[schema.feature_names].copy()

    predictions = np.full(len(input_df), fill_value=np.nan)
    probabilities = np.full(len(input_df), fill_value=np.nan)
    failed_indices: list[int] = []
    failed_reasons: list[str] = []

    for i, row in enumerate(model_input.itertuples(index=False)):
        try:
            row_df = pd.DataFrame([row._asdict()], columns=schema.feature_names)
            pred = pipeline.predict(row_df)
            proba = pipeline.predict_proba(row_df)
            predictions[i] = int(pred[0])
            probabilities[i] = float(proba[0, schema.at_risk_class_index])
        except Exception as exc:
            failed_indices.append(i)
            failed_reasons.append(
                f"Row {i}: inference failed — {exc}"
            )

    # Assemble output preserving all original columns.
    # Use pd.array with Pandas nullable Int64 so that NaN (failed rows) is
    # represented as pd.NA rather than forcing a float column.
    # np.where + np.astype("Int64") is invalid — "Int64" is a Pandas dtype,
    # not a NumPy dtype.
    output = input_df.copy().reset_index(drop=True)
    pred_series = pd.array(
        [int(v) if not np.isnan(v) else pd.NA for v in predictions],
        dtype="Int64",
    )
    output["Predicted_At_Risk"] = pred_series
    output["Estimated_Probability"] = probabilities

    return BatchPredictionResult(
        output_df=output,
        failed_indices=failed_indices,
        failed_reasons=failed_reasons,
    )


def validate_batch_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of missing required model input column names.

    Use this for a fast pre-flight check before calling predict_batch.
    Returns an empty list when all required columns are present.
    """
    schema = get_feature_schema()
    return sorted(set(schema.feature_names) - set(df.columns))


# ── Streamlit-aware cached accessors (import guard) ───────────────────────

def get_streamlit_cached_pipeline():
    """Return a Streamlit @st.cache_resource wrapper around load_saved_model.

    Call this inside a Streamlit page module rather than load_saved_model()
    directly to benefit from Streamlit's resource cache across reruns.

    Usage (in a view module):
        from services.model_service import get_streamlit_cached_pipeline
        pipeline, schema = get_streamlit_cached_pipeline()
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner="Loading risk model…")
        def _cached():
            return load_saved_model()

        return _cached()
    except ImportError:
        # Streamlit not available (e.g., running tests); fall back to module cache
        return load_saved_model()
