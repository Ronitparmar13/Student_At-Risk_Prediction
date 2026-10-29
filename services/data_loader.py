"""Data loader service — CSV ingestion and conservative dataset cleaning.

Cleaning logic is a direct translation of the notebook (Team4_Student_At_Risk.ipynb)
cells 5-8 and the existing app.py clean_dataset() function. Behaviour is preserved
exactly so downstream callers receive the same cleaned DataFrames the original app
produced.

No data is fabricated. If the uploaded file has no valid rows, the returned DataFrame
is empty and the metadata reflects that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Union

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────

TARGET_COL = "At_Risk"
ID_COL = "Student_ID"

# Categorical columns that receive title-case normalisation.
_TITLE_CASE_COLS = ("Gender", "Stream", "Higher_Education_Interest")

# Columns that are never coerced to numeric.
_NON_NUMERIC_SENTINEL = frozenset({ID_COL, "Gender", "Stream", "Higher_Education_Interest"})

# Numeric range rules: (inclusive_low, inclusive_high)
_RANGE_RULES: dict[str, tuple[float, float]] = {}


def _build_range_rules(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Derive range rules from column names — matching the notebook's cleaning cells."""
    rules: dict[str, tuple[float, float]] = {}
    for col in df.columns:
        if "Percentage" in col or col.endswith("_Score"):
            rules[col] = (0.0, 100.0)
    if "Previous_Semester_GPA" in df.columns:
        rules["Previous_Semester_GPA"] = (0.0, 10.0)
    for col in ("Analytical_Thinking", "Creativity_Level",
                "Attention_to_Detail", "Communication_Skills"):
        if col in df.columns:
            rules[col] = (1.0, 10.0)
    if "Age" in df.columns:
        rules["Age"] = (14.0, 100.0)
    # Risk_Score is in range 0-100 but is a leakage column; keep rule for
    # completeness but never feed it to the model.
    if "Risk_Score" in df.columns:
        rules["Risk_Score"] = (0.0, 100.0)
    return rules


# ── Return type ────────────────────────────────────────────────────────────

@dataclass
class CleaningMetadata:
    """Summary of cleaning operations applied to the raw DataFrame."""
    raw_row_count: int = 0
    duplicate_rows_removed: int = 0
    missing_target_rows_dropped: int = 0
    invalid_target_rows_dropped: int = 0
    # {column_name: count_of_values_set_to_NaN}
    out_of_range_counts: dict[str, int] = field(default_factory=dict)
    clean_row_count: int = 0
    has_target: bool = False


# ── Public API ─────────────────────────────────────────────────────────────

def load_and_clean(
    source: Union[str, "os.PathLike[str]", BytesIO, StringIO],
) -> tuple[pd.DataFrame, CleaningMetadata]:
    """Read a CSV from a path or file-like object and apply conservative cleaning.

    Returns
    -------
    (cleaned_df, metadata)
        cleaned_df: the cleaned DataFrame (may be empty if all rows were removed)
        metadata: CleaningMetadata describing what was changed

    Raises
    ------
    ValueError
        If the source cannot be parsed as CSV or the resulting DataFrame has
        zero columns. Callers should surface this as an actionable error.
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if raw.shape[1] == 0:
        raise ValueError("The uploaded file has no columns. Check that it is a valid CSV.")

    cleaned, meta = _clean_dataset(raw)
    return cleaned, meta


def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningMetadata]:
    """Apply conservative cleaning to an already-loaded DataFrame.

    Equivalent to load_and_clean() but accepts a DataFrame directly.
    Useful when the caller has already read the CSV (e.g., from Streamlit's
    file uploader which returns bytes).
    """
    return _clean_dataset(raw.copy())


# ── Internal cleaning implementation ──────────────────────────────────────

def _clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningMetadata]:
    meta = CleaningMetadata(raw_row_count=len(raw))
    df = raw.copy()

    # 1. Normalise column names
    df.columns = df.columns.astype(str).str.strip()

    # 2. Strip leading/trailing whitespace from all string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # 3. Title-case normalisation for known categorical columns
    for col in _TITLE_CASE_COLS:
        if col in df.columns:
            df[col] = df[col].str.title()

    # 4. Coerce non-categorical/non-ID columns to numeric
    non_numeric = _NON_NUMERIC_SENTINEL & set(df.columns)
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Remove exact duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates().copy()
    meta.duplicate_rows_removed = before_dedup - len(df)

    # 6. Apply numeric range rules — out-of-range values → NaN
    range_rules = _build_range_rules(df)
    for col, (low, high) in range_rules.items():
        if col not in df.columns:
            continue
        out_of_range = df[col].notna() & ~df[col].between(low, high)
        count = int(out_of_range.sum())
        if count:
            meta.out_of_range_counts[col] = count
            df.loc[out_of_range, col] = np.nan

    # 7. Handle target column
    meta.has_target = TARGET_COL in df.columns
    if meta.has_target:
        # Mark non-binary target values as NaN (counts separately from missing)
        invalid_target = df[TARGET_COL].notna() & ~df[TARGET_COL].isin([0, 1])
        meta.invalid_target_rows_dropped = int(invalid_target.sum())
        df.loc[invalid_target, TARGET_COL] = np.nan

        missing_target = df[TARGET_COL].isna()
        meta.missing_target_rows_dropped = int(missing_target.sum())
        df = df.dropna(subset=[TARGET_COL]).copy()

        # Ensure int type after cleaning
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    meta.clean_row_count = len(df)
    return df, meta
