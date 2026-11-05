"""Tests for Academic Analyzer"""

import numpy as np
import pandas as pd
import pytest

from ai_study_guide.academic_analyzer import (
    analyze_student_academics,
    to_dict_serializable,
    SUBJECT_COLUMNS,
)


class TestAcademicAnalyzer:
    """Test suite for academic analyzer functionality."""

    def test_correct_class_averages(self):
        """Test that class averages are calculated correctly."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "Science_Score": [85, 75, 95],
        })
        student_record = {
            "Math_Score": 80,
            "Science_Score": 85,
        }

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"]["Math_Score"]["class_average"] == 80.0
        assert result["subject_analysis"]["Science_Score"]["class_average"] == 85.0

    def test_correct_student_to_class_score_gaps(self):
        """Test correct calculation of gaps between student and class average."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],  # average 80
            "English_Score": [60, 80, 100],  # average 80
        })
        student_record = {
            "Math_Score": 90,  # +10 above average
            "English_Score": 70,  # -10 below average
        }

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"]["Math_Score"]["difference"] == 10.0
        assert result["subject_analysis"]["Math_Score"]["performance_status"] == "At or above class average"
        assert result["subject_analysis"]["English_Score"]["difference"] == -10.0
        assert result["subject_analysis"]["English_Score"]["performance_status"] == "Below class average"

    def test_correct_identification_of_strongest_subjects(self):
        """Test identification of strongest subjects."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 80, 80],
            "Science_Score": [80, 80, 80],
            "English_Score": [80, 80, 80],
        })
        student_record = {
            "Math_Score": 95,
            "Science_Score": 85,
            "English_Score": 75,
        }

        result = analyze_student_academics(student_record, class_df)

        assert len(result["strongest_subjects"]) > 0
        # Math should be strongest
        assert result["strongest_subjects"][0]["subject"] == "Math_Score"

    def test_correct_ranking_of_areas_for_improvement(self):
        """Test ranking of areas for improvement by largest negative gap."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 80, 80],
            "Science_Score": [80, 80, 80],
            "English_Score": [80, 80, 80],
        })
        student_record = {
            "Math_Score": 60,  # -20
            "Science_Score": 70,  # -10
            "English_Score": 90,  # +10
        }

        result = analyze_student_academics(student_record, class_df)

        assert len(result["areas_for_improvement"]) == 2
        # Math should be first (largest negative gap)
        assert result["areas_for_improvement"][0]["subject"] == "Math_Score"
        assert result["areas_for_improvement"][0]["difference"] == -20.0
        assert result["areas_for_improvement"][0]["note"] == "area for improvement"

    def test_missing_values(self):
        """Test handling of missing values."""
        class_df = pd.DataFrame({
            "Math_Score": [80, np.nan, 90, 70],
            "Science_Score": [85, 75, 95, 80],
        })
        student_record = {
            "Math_Score": np.nan,
            "Science_Score": 85,
        }

        result = analyze_student_academics(student_record, class_df)

        # Math should have issues due to missing student score
        assert result["subject_analysis"]["Math_Score"]["student_score"] is None
        assert result["subject_analysis"]["Math_Score"]["available"] is False
        # Science should work (class average computed from valid values)
        assert result["subject_analysis"]["Science_Score"]["available"] is True
        assert result["subject_analysis"]["Science_Score"]["class_average"] == 83.75

    def test_missing_columns(self):
        """Test handling of missing columns."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {
            "Math_Score": 85,
            "Science_Score": 90,  # Science not in class_df
        }

        result = analyze_student_academics(student_record, class_df)

        assert "Math_Score" in result["subject_analysis"]
        assert result["subject_analysis"]["Math_Score"]["available"] is True
        # Science might be missing from subject_analysis or marked as unavailable
        if "Science_Score" in result["subject_analysis"]:
            assert result["subject_analysis"]["Science_Score"]["available"] is False
        assert "Science_Score" in result["data_quality"]["subjects_missing"] or \
               len([w for w in result["analysis_warnings"] if "Science_Score" in w]) > 0

    def test_empty_datasets(self):
        """Test handling of empty datasets."""
        class_df = pd.DataFrame()
        student_record = {"Math_Score": 85}

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"] == {}
        assert result["strongest_subjects"] == []
        assert result["areas_for_improvement"] == []
        assert result["overall_subject_average"] is None
        assert result["data_quality"].get("class_dataset_empty") is True
        assert len(result["analysis_warnings"]) > 0

    def test_invalid_and_non_numeric_values(self):
        """Test handling of invalid and non-numeric values."""
        class_df = pd.DataFrame({
            "Math_Score": [80, "invalid", 90, None],
            "Science_Score": [85, 75, 95, 80],
        })
        student_record = {
            "Math_Score": "abc",
            "Science_Score": 85,
        }

        result = analyze_student_academics(student_record, class_df)

        # Math student score is invalid
        assert result["subject_analysis"]["Math_Score"]["student_score"] is None
        assert result["subject_analysis"]["Math_Score"]["available"] is False
        # Class average should be computed from valid numeric values only
        assert result["subject_analysis"]["Math_Score"]["class_average"] == 85.0

    def test_out_of_range_scores(self):
        """Test handling of out of range scores."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {
            "Math_Score": 150,  # Out of range
        }

        result = analyze_student_academics(student_record, class_df)

        # Out of range scores are treated as invalid
        assert result["subject_analysis"]["Math_Score"]["available"] is False
        assert result["subject_analysis"]["Math_Score"]["student_score"] is None
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["student_score_valid"] is False
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["student_score_out_of_range"] is True
        assert len([w for w in result["analysis_warnings"] if "range" in w.lower()]) > 0

    def test_no_identifier_or_sensitive_attributes_in_output(self):
        """Test that sensitive attributes are not in output."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "Student_ID": ["S1", "S2", "S3"],  # Should not affect calculations
            "Gender": ["M", "F", "M"],
            "Risk_Score": [0.5, 0.3, 0.8],
            "At_Risk": [1, 0, 1],
        })
        student_record = {
            "Student_ID": "S1",
            "Gender": "M",
            "Math_Score": 85,
            "Risk_Score": 0.5,
            "At_Risk": 1,
        }

        result = analyze_student_academics(student_record, class_df)

        # Check that sensitive fields are not in the result
        result_str = str(result)
        assert "S1" not in result_str or result_str.count("S1") == 0  # Student ID should not appear
        # More explicitly check structure
        assert "Student_ID" not in result
        assert "Gender" not in result
        assert "Risk_Score" not in result
        assert "At_Risk" not in result
        # Check nested structures
        for key in result:
            if isinstance(result[key], dict):
                assert "Student_ID" not in result[key]
                assert "Gender" not in result[key]

    def test_deterministic_results(self):
        """Test that results are deterministic."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70, 85],
            "Science_Score": [75, 85, 95, 80],
        })
        student_record = {
            "Math_Score": 88,
            "Science_Score": 82,
        }

        result1 = analyze_student_academics(student_record, class_df)
        result2 = analyze_student_academics(student_record, class_df)

        assert result1 == result2

    def test_attendance_and_assignment_completion(self):
        """Test attendance and assignment completion analysis."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "Attendance_Percentage": [85, 90, 75],
            "Assignment_Completion_Percentage": [80, 95, 70],
        })
        student_record = {
            "Math_Score": 85,
            "Attendance_Percentage": 88,
            "Assignment_Completion_Percentage": 75,
        }

        result = analyze_student_academics(student_record, class_df)

        assert result["attendance"]["available"] is True
        assert result["attendance"]["student_value"] == 88.0
        assert result["attendance"]["class_average"] == 83.33
        assert result["assignment_completion"]["available"] is True
        assert result["assignment_completion"]["student_value"] == 75.0

    def test_overall_subject_average(self):
        """Test calculation of overall subject average."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 80, 80],
            "Science_Score": [80, 80, 80],
            "English_Score": [80, 80, 80],
        })
        student_record = {
            "Math_Score": 90,
            "Science_Score": 80,
            "English_Score": 70,
        }

        result = analyze_student_academics(student_record, class_df)

        assert result["overall_subject_average"] == 80.0

    def test_with_series_input(self):
        """Test with pandas Series as student record."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = pd.Series({"Math_Score": 85})

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"]["Math_Score"]["available"] is True
        assert result["subject_analysis"]["Math_Score"]["student_score"] == 85.0

    def test_json_serializable(self):
        """Test that result can be made JSON serializable."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {"Math_Score": 85}

        result = analyze_student_academics(student_record, class_df)
        serializable = to_dict_serializable(result)

        # Should be able to convert to JSON
        import json
        json_str = json.dumps(serializable)
        assert isinstance(json_str, str)

    def test_student_score_above_100(self):
        """Test student score above 100 is treated as invalid."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {"Math_Score": 150}

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"]["Math_Score"]["student_score"] is None
        assert result["subject_analysis"]["Math_Score"]["available"] is False
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["student_score_valid"] is False
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["student_score_out_of_range"] is True
        assert "Math_Score" not in [s["subject"] for s in result["areas_for_improvement"]]
        assert len([w for w in result["analysis_warnings"] if "range" in w.lower()]) > 0

    def test_student_score_below_0(self):
        """Test student score below 0 is treated as invalid."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {"Math_Score": -5}

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"]["Math_Score"]["student_score"] is None
        assert result["subject_analysis"]["Math_Score"]["available"] is False
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["student_score_valid"] is False
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["student_score_out_of_range"] is True

    def test_class_records_containing_out_of_range_scores(self):
        """Test class records containing out of range scores are excluded from averages."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 150, 70, -10],  # 150 and -10 are out of range
        })
        student_record = {"Math_Score": 85}

        result = analyze_student_academics(student_record, class_df)

        # Class average should be calculated from valid values only: (80+70)/2 = 75
        assert result["subject_analysis"]["Math_Score"]["class_average"] == 75.0
        assert result["subject_analysis"]["Math_Score"]["available"] is True
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["class_has_out_of_range"] is True
        assert result["subject_analysis"]["Math_Score"]["difference"] == 10.0  # 85 - 75

    def test_invalid_attendance_and_assignment_percentages(self):
        """Test invalid attendance and assignment percentages."""
        class_df = pd.DataFrame({
            "Attendance_Percentage": [85, 90, 75],
            "Assignment_Completion_Percentage": [80, 95, 70],
        })
        student_record = {
            "Attendance_Percentage": 150,
            "Assignment_Completion_Percentage": -10,
        }

        result = analyze_student_academics(student_record, class_df)

        assert result["attendance"]["available"] is False
        assert result["attendance"]["student_value"] is None
        assert result["attendance"]["data_quality"]["student_value_valid"] is False
        assert result["attendance"]["data_quality"]["student_value_out_of_range"] is True

        assert result["assignment_completion"]["available"] is False
        assert result["assignment_completion"]["student_value"] is None
        assert result["assignment_completion"]["data_quality"]["student_value_valid"] is False
        assert result["assignment_completion"]["data_quality"]["student_value_out_of_range"] is True

    def test_all_values_invalid_for_a_subject(self):
        """Test when all values for a subject are invalid."""
        class_df = pd.DataFrame({
            "Math_Score": [150, -10, 200],
        })
        student_record = {"Math_Score": 85}

        result = analyze_student_academics(student_record, class_df)

        assert result["subject_analysis"]["Math_Score"]["available"] is False
        assert result["subject_analysis"]["Math_Score"]["class_average"] is None
        assert result["subject_analysis"]["Math_Score"]["performance_status"] == "Insufficient data"
        assert result["subject_analysis"]["Math_Score"]["data_quality"]["class_data_valid"] is False
        assert len([w for w in result["analysis_warnings"] if "All values" in w]) > 0

    def test_all_subject_columns(self):
        """Test all subject columns are considered."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "Science_Score": [85, 75, 95],
            "English_Score": [90, 85, 80],
            "Social_Studies_Score": [75, 80, 85],
            "Business_Studies_Score": [88, 92, 86],
            "Computer_Science_Score": [95, 90, 100],
        })
        student_record = {col: 85 for col in SUBJECT_COLUMNS}

        result = analyze_student_academics(student_record, class_df)

        assert len(result["subject_analysis"]) == 6
        for col in SUBJECT_COLUMNS:
            assert col in result["subject_analysis"]
            assert result["subject_analysis"][col]["available"] is True
