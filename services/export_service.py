"""Export service — validated CSV generation for predictions and review queues.

All exports preserve input-row alignment. No rows are silently dropped or
reordered. Failed-inference rows appear in the output with NaN in prediction
columns so the caller can see which rows need attention.

All bytes returned are UTF-8 encoded CSV. The index column is never included.
"""
from __future__ import annotations

import io

import pandas as pd


# ── Public API ─────────────────────────────────────────────────────────────

def build_predictions_csv(
    input_df: pd.DataFrame,
    predictions: "pd.Series | list",
    probabilities: "pd.Series | list | None",
) -> bytes:
    """Build a downloadable CSV from batch prediction results.

    Parameters
    ----------
    input_df:
        The original batch input DataFrame (all original columns preserved).
    predictions:
        Predicted class labels (0 or 1, or NaN for failed rows).
        Must be the same length as input_df.
    probabilities:
        Estimated at-risk probabilities (floats in [0,1], or NaN for failed rows).
        Pass None to omit the probability column from the output.
        Must be the same length as input_df if not None.

    Returns
    -------
    UTF-8 encoded CSV bytes. No index column.

    Raises
    ------
    ValueError
        If predictions length does not match input_df length.
    """
    _check_length("predictions", predictions, len(input_df))
    if probabilities is not None:
        _check_length("probabilities", probabilities, len(input_df))

    output = input_df.copy().reset_index(drop=True)
    output["Predicted_At_Risk"] = list(predictions)

    if probabilities is not None:
        output["Estimated_Probability"] = list(probabilities)

    return _df_to_csv_bytes(output)


def build_review_export(
    flagged_df: pd.DataFrame,
    hide_student_id: bool = True,
) -> bytes:
    """Build a downloadable CSV of the flagged-student review queue.

    Parameters
    ----------
    flagged_df:
        DataFrame of rows the model predicted as at-risk (or a caller-filtered
        subset thereof). Row order is preserved.
    hide_student_id:
        If True (default), Student_ID is removed from the export to minimise
        unnecessary PII in downloaded files. The caller should expose a UI
        option to reveal it when the educator specifically needs it.

    Returns
    -------
    UTF-8 encoded CSV bytes. No index column.
    """
    export = flagged_df.copy().reset_index(drop=True)

    if hide_student_id and "Student_ID" in export.columns:
        export = export.drop(columns=["Student_ID"])

    return _df_to_csv_bytes(export)


def build_quality_report_csv(quality_df: pd.DataFrame) -> bytes:
    """Serialize a data-quality report DataFrame to CSV bytes."""
    return _df_to_csv_bytes(quality_df)


def build_cleaned_csv(cleaned_df: pd.DataFrame) -> bytes:
    """Serialize a cleaned dataset DataFrame to CSV bytes."""
    return _df_to_csv_bytes(cleaned_df)


# ── Internal helpers ───────────────────────────────────────────────────────

def _check_length(name: str, sequence, expected: int) -> None:
    actual = len(sequence) if hasattr(sequence, "__len__") else None
    if actual is not None and actual != expected:
        raise ValueError(
            f"'{name}' has {actual} element(s) but input_df has {expected} row(s). "
            "Predictions and input rows must be in the same order and the same length."
        )


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes without an index column."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")
