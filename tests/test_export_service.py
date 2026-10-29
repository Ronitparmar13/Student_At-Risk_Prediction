"""Unit tests for services/export_service.py."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.export_service import (
    build_cleaned_csv,
    build_predictions_csv,
    build_quality_report_csv,
    build_review_export,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_input(n: int = 5) -> pd.DataFrame:
    """Build a minimal valid DataFrame with exactly n rows for any positive n."""
    # Cycle the At_Risk pattern so any n is supported without length mismatch.
    _pattern = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    at_risk = [_pattern[i % len(_pattern)] for i in range(n)]
    return pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(n)],
        "Age": [20.0] * n,
        "Attendance_Percentage": [75.0] * n,
        "At_Risk": at_risk,
    })


def _parse_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


# ── build_predictions_csv ──────────────────────────────────────────────────

class TestBuildPredictionsCsv:
    def test_returns_bytes(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], [0.3, 0.8, 0.2, 0.9, 0.1])
        assert isinstance(result, bytes)

    def test_output_is_valid_utf8(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], None)
        result.decode("utf-8")  # Should not raise

    def test_output_is_valid_csv(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], [0.3, 0.8, 0.2, 0.9, 0.1])
        parsed = _parse_csv_bytes(result)
        assert isinstance(parsed, pd.DataFrame)

    def test_row_count_matches_input(self):
        df = _make_input(n=7)
        result = build_predictions_csv(df, [0]*7, [0.5]*7)
        parsed = _parse_csv_bytes(result)
        assert len(parsed) == 7

    def test_prediction_column_present(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], None)
        parsed = _parse_csv_bytes(result)
        assert "Predicted_At_Risk" in parsed.columns

    def test_probability_column_present_when_provided(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], [0.3, 0.8, 0.2, 0.9, 0.1])
        parsed = _parse_csv_bytes(result)
        assert "Estimated_Probability" in parsed.columns

    def test_probability_column_absent_when_none(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], None)
        parsed = _parse_csv_bytes(result)
        assert "Estimated_Probability" not in parsed.columns

    def test_no_index_column_in_output(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], None)
        parsed = _parse_csv_bytes(result)
        # A pandas index column would be named 'Unnamed: 0'
        assert "Unnamed: 0" not in parsed.columns

    def test_original_columns_preserved(self):
        df = _make_input()
        result = build_predictions_csv(df, [0, 1, 0, 1, 0], None)
        parsed = _parse_csv_bytes(result)
        for col in df.columns:
            assert col in parsed.columns

    def test_prediction_values_correct(self):
        df = _make_input()
        preds = [1, 0, 1, 0, 1]
        result = build_predictions_csv(df, preds, None)
        parsed = _parse_csv_bytes(result)
        assert list(parsed["Predicted_At_Risk"]) == preds

    def test_row_order_preserved(self):
        """Output row N must correspond to input row N."""
        df = _make_input(n=5)
        df["_original_order"] = list(range(5))
        preds = [1, 0, 1, 0, 1]
        result = build_predictions_csv(df, preds, None)
        parsed = _parse_csv_bytes(result)
        assert list(parsed["_original_order"]) == list(range(5))
        assert list(parsed["Predicted_At_Risk"]) == preds

    def test_raises_on_length_mismatch(self):
        df = _make_input(n=5)
        with pytest.raises(ValueError, match="length"):
            build_predictions_csv(df, [0, 1, 0], None)  # Too few predictions

    def test_nan_predictions_preserved(self):
        """Rows where inference failed should appear as NaN, not be silently dropped."""
        df = _make_input(n=4)
        preds = [0, np.nan, 1, np.nan]
        result = build_predictions_csv(df, preds, None)
        parsed = _parse_csv_bytes(result)
        assert len(parsed) == 4
        # NaN will be read back as NaN by pandas
        assert pd.isna(parsed["Predicted_At_Risk"].iloc[1])
        assert pd.isna(parsed["Predicted_At_Risk"].iloc[3])


# ── build_review_export ────────────────────────────────────────────────────

class TestBuildReviewExport:
    def test_returns_bytes(self):
        df = _make_input()
        result = build_review_export(df)
        assert isinstance(result, bytes)

    def test_row_count_preserved(self):
        df = _make_input(n=6)
        result = build_review_export(df, hide_student_id=False)
        parsed = _parse_csv_bytes(result)
        assert len(parsed) == 6

    def test_student_id_hidden_by_default(self):
        df = _make_input()
        result = build_review_export(df)  # hide_student_id=True by default
        parsed = _parse_csv_bytes(result)
        assert "Student_ID" not in parsed.columns

    def test_student_id_present_when_not_hidden(self):
        df = _make_input()
        result = build_review_export(df, hide_student_id=False)
        parsed = _parse_csv_bytes(result)
        assert "Student_ID" in parsed.columns

    def test_no_index_column_in_output(self):
        df = _make_input()
        result = build_review_export(df)
        parsed = _parse_csv_bytes(result)
        assert "Unnamed: 0" not in parsed.columns

    def test_valid_utf8(self):
        df = _make_input()
        result = build_review_export(df)
        result.decode("utf-8")

    def test_filtered_df_produces_fewer_rows(self):
        df = _make_input(n=10)
        df["Predicted_At_Risk"] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        flagged = df[df["Predicted_At_Risk"] == 1]
        result = build_review_export(flagged, hide_student_id=False)
        parsed = _parse_csv_bytes(result)
        assert len(parsed) == 5


# ── build_quality_report_csv and build_cleaned_csv ────────────────────────

class TestUtilityCsvBuilders:
    def test_quality_report_returns_bytes(self):
        df = pd.DataFrame({"Column": ["A"], "Missing_Count": [0]})
        result = build_quality_report_csv(df)
        assert isinstance(result, bytes)

    def test_cleaned_csv_returns_bytes(self):
        df = _make_input()
        result = build_cleaned_csv(df)
        assert isinstance(result, bytes)

    def test_cleaned_csv_round_trips(self):
        df = _make_input()
        result = build_cleaned_csv(df)
        parsed = _parse_csv_bytes(result)
        assert list(parsed.columns) == list(df.columns)
        assert len(parsed) == len(df)
