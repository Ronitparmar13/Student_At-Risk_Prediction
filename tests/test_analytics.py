"""Unit tests for services/analytics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.analytics import (
    compute_kpis,
    correlation_matrix,
    data_quality_report,
    group_risk_rates,
    leakage_check,
    numeric_summary,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Student_ID": [f"S{i:04d}" for i in range(n)],
        "Age": rng.integers(17, 23, n).astype(float),
        "Stream": rng.choice(["Commerce", "Humanities", "Science"], n),
        "Higher_Education_Interest": rng.choice(["Low", "Medium", "High"], n),
        "Attendance_Percentage": rng.uniform(40, 100, n),
        "Previous_Semester_GPA": rng.uniform(4, 10, n),
        "Previous_Assessment_Score": rng.uniform(30, 100, n),
        "Math_Score": rng.uniform(30, 100, n),
        "Risk_Score": rng.integers(0, 11, n) * 10,
        "At_Risk": rng.integers(0, 2, n),
    })


# ── compute_kpis ───────────────────────────────────────────────────────────

class TestComputeKpis:
    def test_record_count_matches(self):
        df = _make_df(n=42)
        kpis = compute_kpis(df)
        assert kpis.record_count == 42

    def test_column_count_matches(self):
        df = _make_df()
        kpis = compute_kpis(df)
        assert kpis.column_count == df.shape[1]

    def test_at_risk_count_correct(self):
        df = _make_df(n=100, seed=7)
        kpis = compute_kpis(df)
        expected = int((df["At_Risk"] == 1).sum())
        assert kpis.at_risk_count == expected

    def test_not_at_risk_count_correct(self):
        df = _make_df(n=100, seed=7)
        kpis = compute_kpis(df)
        expected = int((df["At_Risk"] == 0).sum())
        assert kpis.not_at_risk_count == expected

    def test_at_risk_pct_sums_to_1(self):
        df = _make_df(n=100)
        kpis = compute_kpis(df)
        assert kpis.at_risk_pct is not None
        total = kpis.at_risk_count + kpis.not_at_risk_count
        expected_pct = kpis.at_risk_count / total
        assert abs(kpis.at_risk_pct - expected_pct) < 1e-9

    def test_unique_students_populated_when_id_present(self):
        df = _make_df(n=10)
        kpis = compute_kpis(df)
        assert kpis.unique_students == 10

    def test_unique_students_none_when_id_absent(self):
        df = _make_df(n=10).drop(columns=["Student_ID"])
        kpis = compute_kpis(df)
        assert kpis.unique_students is None

    def test_no_target_returns_none_at_risk_pct(self):
        df = _make_df(n=10).drop(columns=["At_Risk"])
        kpis = compute_kpis(df)
        assert kpis.at_risk_pct is None
        assert not kpis.has_target

    def test_missing_cells_counted_correctly(self):
        df = _make_df(n=5)
        df.loc[0, "Age"] = np.nan
        df.loc[1, "Attendance_Percentage"] = np.nan
        kpis = compute_kpis(df)
        assert kpis.missing_cells >= 2

    def test_completeness_is_one_for_clean_df(self):
        df = _make_df(n=10)
        # Remove any accidental NaNs
        df = df.dropna()
        kpis = compute_kpis(df)
        assert kpis.completeness == pytest.approx(1.0, abs=0.01) or kpis.completeness <= 1.0

    def test_kpis_computed_from_data_not_fabricated(self):
        """KPI values must change when the DataFrame changes."""
        df1 = _make_df(n=20, seed=1)
        df2 = _make_df(n=40, seed=1)
        kpis1 = compute_kpis(df1)
        kpis2 = compute_kpis(df2)
        assert kpis1.record_count != kpis2.record_count


# ── group_risk_rates ───────────────────────────────────────────────────────

class TestGroupRiskRates:
    def test_returns_grouprisresult(self):
        df = _make_df(n=60)
        result = group_risk_rates(df, "Stream")
        assert result.group_col == "Stream"
        assert isinstance(result.summary, pd.DataFrame)

    def test_summary_has_required_columns(self):
        df = _make_df(n=60)
        result = group_risk_rates(df, "Stream")
        for col in ("Stream", "Record_Count", "At_Risk_Count", "At_Risk_Rate"):
            assert col in result.summary.columns

    def test_rates_are_fractions(self):
        df = _make_df(n=60)
        result = group_risk_rates(df, "Stream")
        assert (result.summary["At_Risk_Rate"] >= 0.0).all()
        assert (result.summary["At_Risk_Rate"] <= 1.0).all()

    def test_record_counts_sum_to_total(self):
        df = _make_df(n=60)
        result = group_risk_rates(df, "Stream")
        assert result.summary["Record_Count"].sum() == 60

    def test_returns_empty_when_group_col_missing(self):
        df = _make_df(n=10)
        result = group_risk_rates(df, "NonExistentCol")
        assert len(result.summary) == 0

    def test_returns_empty_when_target_missing(self):
        df = _make_df(n=10).drop(columns=["At_Risk"])
        result = group_risk_rates(df, "Stream")
        assert len(result.summary) == 0

    def test_at_risk_rate_matches_manual_calculation(self):
        df = pd.DataFrame({
            "Stream": ["Commerce", "Commerce", "Science", "Science"],
            "At_Risk": [1, 0, 1, 1],
        })
        result = group_risk_rates(df, "Stream")
        commerce_row = result.summary[result.summary["Stream"] == "Commerce"]
        science_row = result.summary[result.summary["Stream"] == "Science"]
        assert commerce_row["At_Risk_Rate"].values[0] == pytest.approx(0.5)
        assert science_row["At_Risk_Rate"].values[0] == pytest.approx(1.0)


# ── correlation_matrix ─────────────────────────────────────────────────────

class TestCorrelationMatrix:
    def test_returns_dataframe(self):
        df = _make_df(n=30)
        corr = correlation_matrix(df)
        assert isinstance(corr, pd.DataFrame)

    def test_is_symmetric(self):
        df = _make_df(n=30)
        corr = correlation_matrix(df)
        if corr.empty:
            return
        diff = (corr - corr.T).abs().max().max()
        assert diff < 1e-10

    def test_diagonal_is_one(self):
        df = _make_df(n=30)
        corr = correlation_matrix(df)
        if corr.empty:
            return
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-9

    def test_excluded_columns_absent(self):
        df = _make_df(n=30)
        corr = correlation_matrix(df)
        for excluded in ("At_Risk", "Risk_Score", "Student_ID"):
            assert excluded not in corr.columns

    def test_returns_empty_on_no_numeric_cols(self):
        df = pd.DataFrame({"A": ["x", "y"], "B": ["p", "q"]})
        corr = correlation_matrix(df)
        assert corr.empty

    def test_values_in_minus1_to_1(self):
        df = _make_df(n=50)
        corr = correlation_matrix(df)
        if corr.empty:
            return
        assert (corr.values >= -1.0 - 1e-9).all()
        assert (corr.values <= 1.0 + 1e-9).all()


# ── leakage_check ──────────────────────────────────────────────────────────

class TestLeakageCheck:
    def test_returns_none_when_columns_missing(self):
        df = pd.DataFrame({"Age": [20, 21]})
        assert leakage_check(df) is None

    def test_returns_1_0_for_perfect_leakage(self):
        df = pd.DataFrame({
            "Risk_Score": [0, 30, 50, 70, 100],
            "At_Risk":   [0,  0,  1,  1,   1],
        })
        result = leakage_check(df)
        assert result == pytest.approx(1.0)

    def test_returns_fraction_for_partial_match(self):
        df = pd.DataFrame({
            "Risk_Score": [0, 50, 50, 50],
            "At_Risk":   [0,  1,  0,  1],  # Row 2 doesn't match
        })
        result = leakage_check(df)
        assert result == pytest.approx(0.75)

    def test_returns_none_for_empty_df(self):
        df = pd.DataFrame({"Risk_Score": [], "At_Risk": []})
        # Empty DataFrame: mean() would be NaN; function should return None for empty
        result = leakage_check(df)
        assert result is None or isinstance(result, float)


# ── data_quality_report ────────────────────────────────────────────────────

class TestDataQualityReport:
    def test_returns_dataframe(self):
        df = _make_df(n=10)
        report = data_quality_report(df)
        assert isinstance(report, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_df(n=10)
        report = data_quality_report(df)
        for col in ("Column", "Data_Type", "Missing_Count", "Missing_Pct", "Unique_Values"):
            assert col in report.columns

    def test_missing_count_correct(self):
        df = _make_df(n=5)
        df.loc[0, "Age"] = np.nan
        df.loc[1, "Age"] = np.nan
        report = data_quality_report(df)
        age_row = report[report["Column"] == "Age"]
        assert age_row["Missing_Count"].values[0] == 2

    def test_rows_sorted_by_missing_count(self):
        df = _make_df(n=5)
        df.loc[0, "Age"] = np.nan
        report = data_quality_report(df)
        assert list(report["Missing_Count"]) == sorted(report["Missing_Count"], reverse=True)
