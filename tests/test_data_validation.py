"""Unit tests for services/data_loader.py and services/data_validation.py."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.data_loader import clean_dataframe, load_and_clean
from services.data_validation import (
    DatasetValidationResult,
    validate_batch_csv,
    validate_dataset,
    validate_for_prediction,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_df(n: int = 10, include_target: bool = True, seed: int = 42) -> pd.DataFrame:
    """Minimal valid DataFrame matching the project schema."""
    rng = np.random.default_rng(seed)
    data = {
        "Student_ID": [f"S{i:04d}" for i in range(n)],
        "Age": rng.integers(17, 23, n).astype(float),
        "Gender": rng.choice(["Male", "Female", "Other"], n),
        "Stream": rng.choice(["Commerce", "Humanities", "Science"], n),
        "10th_Percentage": rng.uniform(40, 100, n),
        "12th_Percentage": rng.uniform(40, 100, n),
        "Math_Score": rng.uniform(30, 100, n),
        "Science_Score": rng.uniform(30, 100, n),
        "English_Score": rng.uniform(30, 100, n),
        "Social_Studies_Score": rng.uniform(30, 100, n),
        "Business_Studies_Score": rng.uniform(30, 100, n),
        "Computer_Science_Score": rng.uniform(30, 100, n),
        "Logical_Reasoning_Score": rng.uniform(30, 100, n),
        "Numerical_Aptitude_Score": rng.uniform(30, 100, n),
        "Verbal_Reasoning_Score": rng.uniform(30, 100, n),
        "Spatial_Aptitude_Score": rng.uniform(30, 100, n),
        "Abstract_Reasoning_Score": rng.uniform(30, 100, n),
        "Mechanical_Reasoning_Score": rng.uniform(30, 100, n),
        "Perceptual_Speed_Score": rng.uniform(30, 100, n),
        "Analytical_Thinking": rng.integers(1, 11, n).astype(float),
        "Creativity_Level": rng.integers(1, 11, n).astype(float),
        "Attention_to_Detail": rng.integers(1, 11, n).astype(float),
        "Communication_Skills": rng.integers(1, 11, n).astype(float),
        "Higher_Education_Interest": rng.choice(["Low", "Medium", "High"], n),
        "Attendance_Percentage": rng.uniform(50, 100, n),
        "Assignment_Completion_Percentage": rng.uniform(50, 100, n),
        "Previous_Semester_GPA": rng.uniform(4, 10, n),
        "Previous_Assessment_Score": rng.uniform(30, 100, n),
        "Risk_Score": rng.integers(0, 11, n) * 10,
    }
    if include_target:
        data["At_Risk"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


# ── data_loader tests ──────────────────────────────────────────────────────

class TestLoadAndClean:
    def test_returns_dataframe_and_metadata(self):
        df = _make_df()
        csv_bytes = io.BytesIO(df.to_csv(index=False).encode())
        cleaned, meta = load_and_clean(csv_bytes)
        assert isinstance(cleaned, pd.DataFrame)
        assert meta.raw_row_count == 10

    def test_invalid_csv_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not read"):
            load_and_clean(io.BytesIO(b"not,a,valid\x00csv\xff"))

    def test_empty_csv_raises_value_error(self):
        with pytest.raises(ValueError):
            load_and_clean(io.BytesIO(b""))


class TestCleanDataframe:
    def test_duplicate_rows_removed(self):
        df = _make_df(n=5)
        doubled = pd.concat([df, df], ignore_index=True)
        cleaned, meta = clean_dataframe(doubled)
        assert meta.duplicate_rows_removed == 5
        assert len(cleaned) == 5

    def test_out_of_range_percentage_set_to_nan(self):
        df = _make_df(n=5)
        df.loc[0, "Attendance_Percentage"] = 150.0  # Out of range
        cleaned, meta = clean_dataframe(df)
        assert pd.isna(cleaned.loc[0, "Attendance_Percentage"])
        assert meta.out_of_range_counts.get("Attendance_Percentage", 0) >= 1

    def test_out_of_range_gpa_set_to_nan(self):
        df = _make_df(n=5)
        df.loc[1, "Previous_Semester_GPA"] = 11.0  # GPA > 10
        cleaned, meta = clean_dataframe(df)
        assert pd.isna(cleaned.loc[1, "Previous_Semester_GPA"])

    def test_out_of_range_soft_skill_set_to_nan(self):
        df = _make_df(n=5)
        df.loc[2, "Analytical_Thinking"] = 11.0  # > max of 10
        cleaned, meta = clean_dataframe(df)
        assert pd.isna(cleaned.loc[2, "Analytical_Thinking"])

    def test_missing_target_rows_dropped(self):
        df = _make_df(n=5)
        df.loc[0, "At_Risk"] = np.nan
        cleaned, meta = clean_dataframe(df)
        assert len(cleaned) == 4
        assert meta.missing_target_rows_dropped == 1

    def test_invalid_target_values_dropped(self):
        df = _make_df(n=5)
        df.loc[0, "At_Risk"] = 5  # Not 0 or 1
        cleaned, meta = clean_dataframe(df)
        assert len(cleaned) == 4
        assert meta.invalid_target_rows_dropped == 1

    def test_target_column_is_int_after_cleaning(self):
        df = _make_df(n=5)
        cleaned, _ = clean_dataframe(df)
        assert cleaned["At_Risk"].dtype in (int, np.int64, np.int32)

    def test_metadata_clean_row_count_is_accurate(self):
        df = _make_df(n=10)
        df.loc[0, "At_Risk"] = np.nan  # Will be dropped
        cleaned, meta = clean_dataframe(df)
        assert meta.clean_row_count == len(cleaned)
        assert meta.clean_row_count == 9

    def test_column_name_whitespace_stripped(self):
        df = _make_df(n=3)
        df.columns = [f" {c} " for c in df.columns]  # Add spaces
        cleaned, _ = clean_dataframe(df)
        for col in cleaned.columns:
            assert col == col.strip(), f"Column has whitespace: '{col}'"

    def test_categorical_title_cased(self):
        df = _make_df(n=3)
        df["Stream"] = "commerce"  # Lowercase
        df["Gender"] = "MALE"      # Uppercase
        cleaned, _ = clean_dataframe(df)
        assert cleaned["Stream"].iloc[0] == "Commerce"
        assert cleaned["Gender"].iloc[0] == "Male"

    def test_no_target_column_handled_gracefully(self):
        df = _make_df(n=5, include_target=False)
        cleaned, meta = clean_dataframe(df)
        assert not meta.has_target
        assert len(cleaned) == 5  # No rows dropped without a target

    def test_raw_row_count_matches_input(self):
        df = _make_df(n=20)
        _, meta = clean_dataframe(df)
        assert meta.raw_row_count == 20


# ── data_validation tests ──────────────────────────────────────────────────

class TestValidateDataset:
    def test_valid_df_has_no_errors(self):
        df = _make_df(n=20)
        cleaned, _ = clean_dataframe(df)
        result = validate_dataset(cleaned)
        assert result.is_valid

    def test_empty_df_produces_error(self):
        result = validate_dataset(pd.DataFrame())
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_none_produces_error(self):
        result = validate_dataset(None)
        assert not result.is_valid

    def test_missing_target_column_produces_error(self):
        df = _make_df(n=5, include_target=False)
        cleaned, _ = clean_dataframe(df)
        result = validate_dataset(cleaned)
        assert not result.is_valid
        assert any("At_Risk" in e for e in result.errors)

    def test_leakage_match_produces_warning(self):
        df = _make_df(n=10)
        # Force perfect match
        df["At_Risk"] = (df["Risk_Score"] >= 50).astype(int)
        cleaned, _ = clean_dataframe(df)
        result = validate_dataset(cleaned)
        assert any("Risk_Score" in w for w in result.warnings)

    def test_small_dataset_produces_warning(self):
        df = _make_df(n=3)
        cleaned, _ = clean_dataframe(df)
        result = validate_dataset(cleaned)
        assert any("3" in w for w in result.warnings)

    def test_result_is_valid_property(self):
        result = DatasetValidationResult(errors=["something wrong"])
        assert not result.is_valid
        result2 = DatasetValidationResult()
        assert result2.is_valid


class TestValidateBatchCsv:
    def test_valid_batch_no_errors(self):
        from services.model_service import get_feature_schema
        schema = get_feature_schema()
        batch = pd.DataFrame(columns=schema.feature_names)
        batch.loc[0] = 50.0
        result = validate_batch_csv(batch, schema.feature_names)
        assert result.is_valid

    def test_missing_columns_reported(self):
        from services.model_service import get_feature_schema
        schema = get_feature_schema()
        # Drop first 3 required columns
        dropped = schema.feature_names[:3]
        remaining = schema.feature_names[3:]
        batch = pd.DataFrame(columns=remaining)
        batch.loc[0] = 50.0
        result = validate_batch_csv(batch, schema.feature_names)
        assert not result.is_valid
        for col in dropped:
            assert col in result.missing_required_cols

    def test_empty_df_produces_error(self):
        result = validate_batch_csv(pd.DataFrame(), ["col1", "col2"])
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_extra_columns_reported_as_warnings(self):
        from services.model_service import get_feature_schema
        schema = get_feature_schema()
        batch = pd.DataFrame(columns=schema.feature_names + ["Extra_Col"])
        batch.loc[0] = 50.0
        result = validate_batch_csv(batch, schema.feature_names)
        assert result.is_valid  # Extra columns are not blocking errors
        assert "Extra_Col" in result.extra_cols

    def test_forbidden_columns_produce_warning_not_error(self):
        from services.model_service import get_feature_schema
        schema = get_feature_schema()
        batch = pd.DataFrame(columns=schema.feature_names + ["Risk_Score"])
        batch.loc[0] = 50.0
        result = validate_batch_csv(batch, schema.feature_names)
        assert result.is_valid
        assert any("Risk_Score" in w for w in result.warnings)
