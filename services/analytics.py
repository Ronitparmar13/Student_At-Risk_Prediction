"""Analytics service — KPI computation, group summaries, and correlation matrices.

All figures are derived exclusively from the DataFrame passed by the caller.
No values are fabricated, estimated, or hard-coded. If the required columns
are absent, the affected metric is omitted from the result rather than
substituted with a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET_COL = "At_Risk"
_LEAKAGE_COL = "Risk_Score"
_ID_COL = "Student_ID"

# Columns excluded from the numeric distribution and correlation analysis
# because they are identifiers, leakage features, or the target itself.
_EXCLUDE_FROM_ANALYSIS = frozenset({TARGET_COL, _LEAKAGE_COL, _ID_COL})


# ── Return types ───────────────────────────────────────────────────────────

@dataclass
class KPIResult:
    """Top-line metrics computed from a cleaned dataset."""
    record_count: int = 0
    column_count: int = 0
    missing_cells: int = 0
    duplicate_rows: int = 0
    has_target: bool = False
    # Only populated when At_Risk is present
    at_risk_count: int | None = None
    not_at_risk_count: int | None = None
    at_risk_pct: float | None = None          # 0.0 – 1.0
    # Only populated when Student_ID is present
    unique_students: int | None = None
    # Data quality: fraction of non-missing cells (0.0 – 1.0)
    completeness: float = 1.0


@dataclass
class GroupRiskResult:
    """At-risk rate summary grouped by a categorical column."""
    group_col: str = ""
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    # summary columns: [group_col, "Record_Count", "At_Risk_Count", "At_Risk_Rate"]
    # At_Risk_Rate is a fraction 0.0–1.0


# ── Public API ─────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> KPIResult:
    """Compute top-line KPIs from a cleaned dataset.

    All values come from the supplied DataFrame. Returns a KPIResult with
    None for any metric that cannot be computed from the available columns.
    """
    result = KPIResult(
        record_count=len(df),
        column_count=df.shape[1],
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
    )

    total_cells = df.shape[0] * df.shape[1]
    result.completeness = (
        float(1.0 - df.isna().sum().sum() / total_cells)
        if total_cells > 0 else 1.0
    )

    if TARGET_COL in df.columns:
        result.has_target = True
        target_series = df[TARGET_COL].dropna()
        result.at_risk_count = int((target_series == 1).sum())
        result.not_at_risk_count = int((target_series == 0).sum())
        total_valid = result.at_risk_count + result.not_at_risk_count
        result.at_risk_pct = (
            float(result.at_risk_count / total_valid)
            if total_valid > 0 else None
        )

    if _ID_COL in df.columns:
        result.unique_students = int(df[_ID_COL].nunique(dropna=True))

    return result


def group_risk_rates(df: pd.DataFrame, group_col: str) -> GroupRiskResult:
    """Compute at-risk rates grouped by a categorical column.

    Parameters
    ----------
    df:
        Cleaned DataFrame containing At_Risk and the specified group_col.
    group_col:
        Name of the categorical column to group by.

    Returns
    -------
    GroupRiskResult with a summary DataFrame.
    If At_Risk or group_col is absent, returns an empty result.
    """
    result = GroupRiskResult(group_col=group_col)

    if TARGET_COL not in df.columns or group_col not in df.columns:
        return result

    grp = (
        df.groupby(group_col, dropna=False)[TARGET_COL]
        .agg(
            Record_Count="count",
            At_Risk_Count="sum",
        )
        .reset_index()
    )
    grp["At_Risk_Rate"] = grp["At_Risk_Count"] / grp["Record_Count"]
    grp["At_Risk_Rate"] = grp["At_Risk_Rate"].round(4)

    # Sort by at-risk rate descending for easier reading
    grp = grp.sort_values("At_Risk_Rate", ascending=False).reset_index(drop=True)

    result.summary = grp
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for numeric columns.

    Excludes At_Risk, Risk_Score, and Student_ID from the analysis.
    Returns an empty DataFrame if fewer than 2 numeric columns remain.

    Parameters
    ----------
    df:
        Cleaned DataFrame.

    Returns
    -------
    Symmetric correlation DataFrame; columns and index are feature names.
    """
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    analysis_cols = [c for c in numeric if c not in _EXCLUDE_FROM_ANALYSIS]

    if len(analysis_cols) < 2:
        return pd.DataFrame()

    return df[analysis_cols].corr(method="pearson")


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for analysis-eligible numeric columns.

    Excludes At_Risk, Risk_Score, and Student_ID.

    Returns a DataFrame indexed by column name with columns:
    count, mean, std, min, 25%, 50%, 75%, max, missing_count, missing_pct.
    """
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    cols = [c for c in numeric if c not in _EXCLUDE_FROM_ANALYSIS]

    if not cols:
        return pd.DataFrame()

    desc = df[cols].describe().T
    desc["missing_count"] = df[cols].isna().sum()
    desc["missing_pct"] = (df[cols].isna().mean() * 100).round(2)
    return desc.round(3)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column data quality summary.

    Columns: Column, Data_Type, Missing_Count, Missing_Pct, Unique_Values.
    Sorted by Missing_Count descending.
    """
    report = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": [str(df[c].dtype) for c in df.columns],
        "Missing_Count": [int(df[c].isna().sum()) for c in df.columns],
        "Missing_Pct": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
        "Unique_Values": [int(df[c].nunique(dropna=False)) for c in df.columns],
    })
    return report.sort_values("Missing_Count", ascending=False).reset_index(drop=True)


def leakage_check(df: pd.DataFrame) -> float | None:
    """Return the fraction of rows where Risk_Score >= 50 matches At_Risk.

    Returns None if either column is absent. Returns a float in [0.0, 1.0]
    otherwise. A value of 1.0 indicates a perfect match — a strong (but not
    conclusive) indicator that the target may be rule-generated.

    Callers must present this as an advisory, not proof of label construction.
    """
    if {_LEAKAGE_COL, TARGET_COL}.issubset(df.columns) and len(df) > 0:
        derived = (df[_LEAKAGE_COL] >= 50).astype(int)
        return float((derived == df[TARGET_COL]).mean())
    return None
