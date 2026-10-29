"""Data validation service — schema checks and actionable error messages.

Returns structured lists of error/warning strings so the UI layer can
display them without knowing what exactly went wrong. Never raises exceptions
for data-quality issues; those are surfaced as messages in the return value.

Only ModelInferenceError-class structural failures are re-raised.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TARGET_COL = "At_Risk"


# ── Return types ───────────────────────────────────────────────────────────

@dataclass
class DatasetValidationResult:
    """Result of validating an uploaded dataset."""
    errors: list[str] = field(default_factory=list)      # Blocking problems
    warnings: list[str] = field(default_factory=list)    # Non-blocking advisories

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class BatchValidationResult:
    """Result of validating a batch prediction CSV."""
    missing_required_cols: list[str] = field(default_factory=list)
    extra_cols: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.missing_required_cols) == 0


# ── Dataset validation ─────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> DatasetValidationResult:
    """Validate a cleaned dataset for use in analytics and display.

    Checks:
    - At least one row of data
    - Target column present and binary
    - Both target classes present (required for meaningful analysis)
    - Leakage advisory if Risk_Score matches At_Risk exactly

    Parameters
    ----------
    df:
        A cleaned DataFrame (output of data_loader.clean_dataframe or
        data_loader.load_and_clean).

    Returns
    -------
    DatasetValidationResult with errors and warnings lists.
    """
    result = DatasetValidationResult()

    if df is None or len(df) == 0:
        result.errors.append(
            "The dataset is empty after cleaning. "
            "Upload a CSV with at least one valid student record."
        )
        return result

    # Target column checks
    if TARGET_COL not in df.columns:
        result.errors.append(
            f"Required target column '{TARGET_COL}' is missing. "
            "Ensure the CSV has a column named exactly 'At_Risk' with values 0 and 1."
        )
    else:
        unique_vals = set(df[TARGET_COL].dropna().unique())
        non_binary = unique_vals - {0, 1}
        if non_binary:
            result.errors.append(
                f"'{TARGET_COL}' contains unexpected values: {sorted(non_binary)}. "
                "Only 0 (not at risk) and 1 (at risk) are accepted."
            )
        elif unique_vals == {0}:
            result.warnings.append(
                "All records are class 0 (not at risk). "
                "Class 1 is absent — group analysis will reflect only one class."
            )
        elif unique_vals == {1}:
            result.warnings.append(
                "All records are class 1 (at risk). "
                "Class 0 is absent — group analysis will reflect only one class."
            )

    # Leakage advisory
    if {"Risk_Score", TARGET_COL}.issubset(df.columns) and len(df) > 0:
        derived = (df["Risk_Score"] >= 50).astype(int)
        match_rate = float((derived == df[TARGET_COL]).mean())
        if match_rate == 1.0:
            result.warnings.append(
                "Advisory: Risk_Score >= 50 matches At_Risk for every row in this dataset. "
                "This suggests the target may have been rule-generated from Risk_Score. "
                "Provenance is unverified. Risk_Score is excluded from model inputs."
            )
        elif match_rate >= 0.95:
            result.warnings.append(
                f"Advisory: Risk_Score >= 50 matches At_Risk for {match_rate:.1%} of rows. "
                "This may indicate the target was partially rule-generated. "
                "Risk_Score is excluded from model inputs."
            )

    # Row count advisory
    if len(df) < 10:
        result.warnings.append(
            f"Only {len(df)} row(s) after cleaning. "
            "Analytics may not be meaningful with very few records."
        )

    return result


def validate_for_prediction(df: pd.DataFrame, required_cols: list[str]) -> DatasetValidationResult:
    """Validate that a DataFrame has the columns required for batch prediction.

    Parameters
    ----------
    df:
        The batch upload DataFrame (uncleaned is fine — we only check columns here).
    required_cols:
        The list of model input feature names from FeatureSchema.feature_names.

    Returns
    -------
    DatasetValidationResult
    """
    result = DatasetValidationResult()

    if df is None or len(df) == 0:
        result.errors.append(
            "The uploaded batch file is empty. "
            "Upload a CSV with at least one student record."
        )
        return result

    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        result.errors.append(
            f"Batch CSV is missing {len(missing)} required model input column(s): "
            f"{missing}. "
            "All 26 model input features must be present. "
            "Check the column names match exactly (case-sensitive)."
        )

    # Warn if forbidden columns are present (they will be stripped before inference)
    from services.model_service import _FORBIDDEN_COLS  # noqa: PLC0415
    found_forbidden = sorted(_FORBIDDEN_COLS & set(df.columns))
    if found_forbidden:
        result.warnings.append(
            f"Column(s) {found_forbidden} will be carried through to the output "
            "but will not be used as model inputs."
        )

    return result


# ── Batch CSV validation (pre-inference) ──────────────────────────────────

def validate_batch_csv(
    df: pd.DataFrame,
    required_cols: list[str],
) -> BatchValidationResult:
    """Validate a batch prediction CSV against the required model feature list.

    Parameters
    ----------
    df:
        The raw batch upload DataFrame.
    required_cols:
        The list of model input feature names from FeatureSchema.feature_names.

    Returns
    -------
    BatchValidationResult
    """
    result = BatchValidationResult()

    if df is None or len(df) == 0:
        result.errors.append(
            "The batch CSV is empty. "
            "Upload a file with at least one student record to run predictions."
        )
        return result

    missing = sorted(set(required_cols) - set(df.columns))
    result.missing_required_cols = missing
    if missing:
        result.errors.append(
            f"Missing {len(missing)} required column(s): {missing}. "
            "All 26 model input features must be present."
        )

    # Report any forbidden columns that will be stripped
    from services.model_service import _FORBIDDEN_COLS  # noqa: PLC0415
    found_forbidden = sorted(_FORBIDDEN_COLS & set(df.columns))
    if found_forbidden:
        result.warnings.append(
            f"{found_forbidden} will be included in output columns for reference "
            "but will not be passed to the model."
        )

    # Extra columns that are not model features (informational)
    extra = sorted(
        set(df.columns) - set(required_cols) - set(found_forbidden)
    )
    result.extra_cols = extra
    if extra:
        result.warnings.append(
            f"{len(extra)} extra column(s) in the file will be passed through "
            f"to the output unchanged: {extra}."
        )

    return result
