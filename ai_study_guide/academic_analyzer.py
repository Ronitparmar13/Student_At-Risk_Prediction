"""Academic Analyzer for Student Success & At-Risk Prediction.

This module provides deterministic analysis of a student's academic performance
against the available class dataset. It is designed to provide verified academic
context to AI services without exposing sensitive student information or
modifying existing application logic.
"""

from typing import Any, Dict, List, Optional, Union

import math
import numpy as np
import pandas as pd


SUBJECT_COLUMNS = [
    "Math_Score",
    "Science_Score",
    "English_Score",
    "Social_Studies_Score",
    "Business_Studies_Score",
    "Computer_Science_Score",
]

INDICATOR_COLUMNS = {
    "attendance": "Attendance_Percentage",
    "assignment_completion": "Assignment_Completion_Percentage",
    "previous_semester_gpa": "Previous_Semester_GPA",
    "previous_assessment_score": "Previous_Assessment_Score",
}

SENSITIVE_COLUMNS = {
    "Student_ID",
    "Gender",
    "Risk_Score",
    "At_Risk",
    "Email",
    "Name",
    "Student_Name",
}


def _is_numeric_series_valid(series: pd.Series) -> bool:
    """Check if a series contains numeric values that can be meaningfully used."""
    if series is None or len(series) == 0:
        return False
    # Try to convert to numeric
    numeric_series = pd.to_numeric(series, errors="coerce")
    # Check if there are any non-NaN values
    return numeric_series.notna().sum() > 0


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None if invalid."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        # Handle numpy types
        if isinstance(value, (np.integer, np.floating)):
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        # Try to convert
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None


def _validate_score_range(score: float) -> bool:
    """Check if score is within expected 0-100 range."""
    if score is None:
        return False
    return 0.0 <= score <= 100.0


def analyze_student_academics(
    student_record: Union[pd.Series, Dict[str, Any]],
    class_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Analyze a selected student's academic performance against the available class dataset.

    Args:
        student_record: A pandas Series or dict containing the student's academic data.
                       Sensitive identifiers (Student_ID, Gender, Risk_Score, At_Risk, etc.)
                       must not be used for calculations and will not be included in output.
        class_df: DataFrame containing all students' data for class comparisons.
                  Sensitive identifiers must not be used for calculations.

    Returns:
        A dictionary containing:
        - subject_analysis: Dict with per-subject analysis
        - strongest_subjects: List of strongest subjects
        - areas_for_improvement: List of subjects below class average
        - overall_subject_average: Overall average across available subjects for student
        - attendance: Attendance analysis
        - assignment_completion: Assignment completion analysis
        - data_quality: Information about data quality and availability
        - analysis_warnings: List of warnings about data issues
    """
    analysis_warnings: List[str] = []
    data_quality: Dict[str, Any] = {
        "subjects_analyzed": 0,
        "subjects_missing": [],
        "subjects_invalid": [],
        "student_record_valid": True,
    }

    # Convert student_record to a dict-like structure for easier access
    if isinstance(student_record, pd.Series):
        student_dict = student_record.to_dict()
    elif isinstance(student_record, dict):
        student_dict = dict(student_record)
    else:
        analysis_warnings.append("Invalid student record type")
        data_quality["student_record_valid"] = False
        return _build_empty_result(analysis_warnings, data_quality)

    # Validate inputs
    if class_df is None or (isinstance(class_df, pd.DataFrame) and class_df.empty):
        analysis_warnings.append("Empty or None class dataset provided")
        data_quality["class_dataset_empty"] = True
        return _build_empty_result(analysis_warnings, data_quality)

    if not isinstance(class_df, pd.DataFrame):
        analysis_warnings.append("Invalid class dataset type")
        return _build_empty_result(analysis_warnings, data_quality)

    # Analyze each subject
    subject_analysis: Dict[str, Dict[str, Any]] = {}
    student_subject_scores: List[float] = []

    for subject in SUBJECT_COLUMNS:
        subject_result = _analyze_subject(subject, student_dict, class_df, analysis_warnings)
        if subject_result is not None:
            subject_analysis[subject] = subject_result
            if subject_result.get("student_score") is not None and subject_result.get("available", False):
                student_subject_scores.append(subject_result["student_score"])
        else:
            data_quality["subjects_missing"].append(subject)

    data_quality["subjects_analyzed"] = len([s for s in subject_analysis.values() if s.get("available", False)])

    # Identify strongest subjects and areas for improvement
    strongest_subjects = []
    areas_for_improvement = []

    available_subjects = [
        (subject, analysis) for subject, analysis in subject_analysis.items()
        if analysis.get("available", False) and analysis.get("student_score") is not None
    ]

    if available_subjects:
        # Sort by student score descending for strongest
        sorted_by_score = sorted(available_subjects, key=lambda x: x[1]["student_score"], reverse=True)
        # Strongest subjects - top performers (all available subjects that are at or above average or just highest)
        for subject, analysis in sorted_by_score:
            if analysis.get("performance_status") in ["At or above class average", "Below class average"]:
                strongest_subjects.append({
                    "subject": subject,
                    "student_score": analysis["student_score"],
                    "class_average": analysis["class_average"],
                    "difference": analysis["difference"],
                    "performance_status": analysis["performance_status"],
                })

        # Areas for improvement - subjects below class average, ranked by largest negative gap
        below_average = [
            (subject, analysis) for subject, analysis in available_subjects
            if analysis.get("performance_status") == "Below class average"
        ]
        if below_average:
            sorted_by_gap = sorted(below_average, key=lambda x: x[1]["difference"])  # More negative first
            for subject, analysis in sorted_by_gap:
                areas_for_improvement.append({
                    "subject": subject,
                    "student_score": analysis["student_score"],
                    "class_average": analysis["class_average"],
                    "difference": analysis["difference"],
                    "performance_status": analysis["performance_status"],
                    "note": "area for improvement",
                })

    # Calculate overall available subject average
    overall_subject_average = None
    if student_subject_scores:
        overall_subject_average = round(sum(student_subject_scores) / len(student_subject_scores), 2)

    # Analyze attendance
    attendance_analysis = _analyze_indicator(
        "attendance",
        INDICATOR_COLUMNS["attendance"],
        student_dict,
        class_df,
        analysis_warnings
    )

    # Analyze assignment completion
    assignment_analysis = _analyze_indicator(
        "assignment_completion",
        INDICATOR_COLUMNS["assignment_completion"],
        student_dict,
        class_df,
        analysis_warnings,
    )

    result = {
        "subject_analysis": subject_analysis,
        "strongest_subjects": strongest_subjects,
        "areas_for_improvement": areas_for_improvement,
        "overall_subject_average": overall_subject_average,
        "attendance": attendance_analysis,
        "assignment_completion": assignment_analysis,
        "data_quality": data_quality,
        "analysis_warnings": analysis_warnings,
    }

    return result


def _analyze_subject(
    subject: str,
    student_dict: Dict[str, Any],
    class_df: pd.DataFrame,
    analysis_warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """Analyze a single subject."""
    result = {
        "subject": subject,
        "student_score": None,
        "class_average": None,
        "difference": None,
        "performance_status": "Insufficient data",
        "available": False,
        "data_quality": {
            "student_score_available": False,
            "class_data_available": False,
            "student_score_valid": False,
            "class_data_valid": False,
            "student_score_out_of_range": False,
            "class_has_out_of_range": False,
        },
    }

    # Check if subject column exists in class_df
    if subject not in class_df.columns:
        analysis_warnings.append(f"Subject column '{subject}' not found in class dataset")
        result["data_quality"]["class_data_available"] = False
        return result

    # Analyze class data
    class_series = class_df[subject].copy()
    # Convert to numeric, coercing errors
    class_numeric = pd.to_numeric(class_series, errors="coerce")
    # Filter out NaN
    class_numeric_no_nan = class_numeric.dropna()

    if len(class_numeric_no_nan) == 0:
        analysis_warnings.append(f"No valid numeric data for '{subject}' in class dataset")
        result["data_quality"]["class_data_valid"] = False
        return result

    # Check for out of range values in class - exclude them from averages
    out_of_range_class_mask = (class_numeric_no_nan < 0) | (class_numeric_no_nan > 100)
    out_of_range_class = class_numeric_no_nan[out_of_range_class_mask]
    if len(out_of_range_class) > 0:
        analysis_warnings.append(f"Class data for '{subject}' contains out of range values (0-100 expected); excluded from class average")
        result["data_quality"]["class_has_out_of_range"] = True

    valid_class_values = class_numeric_no_nan[~out_of_range_class_mask]

    if len(valid_class_values) == 0:
        analysis_warnings.append(f"All values for '{subject}' in class dataset are invalid or out of range")
        result["data_quality"]["class_data_valid"] = False
        return result

    class_average = float(valid_class_values.mean())
    result["class_average"] = round(class_average, 2)
    result["data_quality"]["class_data_available"] = True
    result["data_quality"]["class_data_valid"] = True

    # Analyze student score
    if subject not in student_dict:
        result["data_quality"]["student_score_available"] = False
        return result

    student_raw = student_dict[subject]
    result["data_quality"]["student_score_available"] = True

    student_score = _safe_float(student_raw)
    if student_score is None:
        analysis_warnings.append(f"Invalid or non-numeric score for student in '{subject}'")
        result["data_quality"]["student_score_valid"] = False
        return result

    # Check if score is in range - if not, treat as invalid
    if not _validate_score_range(student_score):
        analysis_warnings.append(f"Student score for '{subject}' is outside expected range 0-100; excluded from analysis")
        result["data_quality"]["student_score_valid"] = False
        result["data_quality"]["student_score_out_of_range"] = True
        return result

    result["data_quality"]["student_score_valid"] = True

    result["student_score"] = round(student_score, 2)

    # Calculate difference from class average
    difference = student_score - class_average
    result["difference"] = round(difference, 2)

    # Determine performance status
    if abs(difference) < 1e-9:  # At or very close to average (considering floating point)
        result["performance_status"] = "At or above class average"
    elif difference >= 0:
        result["performance_status"] = "At or above class average"
    else:
        result["performance_status"] = "Below class average"

    result["available"] = True

    return result


def _analyze_indicator(
    indicator_name: str,
    column_name: str,
    student_dict: Dict[str, Any],
    class_df: pd.DataFrame,
    analysis_warnings: List[str],
) -> Dict[str, Any]:
    """Analyze attendance or assignment completion indicators."""
    result = {
        "indicator": indicator_name,
        "column_name": column_name,
        "student_value": None,
        "class_average": None,
        "difference": None,
        "available": False,
        "data_quality": {
            "student_value_available": False,
            "class_data_available": False,
            "student_value_valid": False,
            "class_data_valid": False,
            "student_value_out_of_range": False,
            "class_has_out_of_range": False,
        },
    }

    # Check class data
    if column_name not in class_df.columns:
        analysis_warnings.append(f"Indicator column '{column_name}' not found in class dataset")
        result["data_quality"]["class_data_available"] = False
        return result

    class_series = class_df[column_name].copy()
    class_numeric = pd.to_numeric(class_series, errors="coerce")
    class_numeric_no_nan = class_numeric.dropna()

    if len(class_numeric_no_nan) == 0:
        analysis_warnings.append(f"No valid numeric data for '{column_name}' in class dataset")
        result["data_quality"]["class_data_valid"] = False
        return result

    # Check for out of range values in class - exclude them from averages
    out_of_range_class_mask = (class_numeric_no_nan < 0) | (class_numeric_no_nan > 100)
    out_of_range_class = class_numeric_no_nan[out_of_range_class_mask]
    if len(out_of_range_class) > 0:
        analysis_warnings.append(f"Class data for '{column_name}' contains out of range values (0-100 expected); excluded from class average")
        result["data_quality"]["class_has_out_of_range"] = True

    valid_class_values = class_numeric_no_nan[~out_of_range_class_mask]

    if len(valid_class_values) == 0:
        analysis_warnings.append(f"All values for '{column_name}' in class dataset are invalid or out of range")
        result["data_quality"]["class_data_valid"] = False
        return result

    class_average = float(valid_class_values.mean())
    result["class_average"] = round(class_average, 2)
    result["data_quality"]["class_data_available"] = True
    result["data_quality"]["class_data_valid"] = True

    # Check student data
    if column_name not in student_dict:
        result["data_quality"]["student_value_available"] = False
        return result

    student_raw = student_dict[column_name]
    result["data_quality"]["student_value_available"] = True

    student_value = _safe_float(student_raw)
    if student_value is None:
        analysis_warnings.append(f"Invalid or non-numeric value for student in '{column_name}'")
        result["data_quality"]["student_value_valid"] = False
        return result

    # Check if value is in range - if not, treat as invalid
    if not _validate_score_range(student_value):
        analysis_warnings.append(f"Student value for '{column_name}' is outside expected range 0-100; excluded from analysis")
        result["data_quality"]["student_value_valid"] = False
        result["data_quality"]["student_value_out_of_range"] = True
        return result

    result["data_quality"]["student_value_valid"] = True

    result["student_value"] = round(student_value, 2)
    difference = student_value - class_average
    result["difference"] = round(difference, 2)
    result["available"] = True

    return result


def _build_empty_result(
    analysis_warnings: List[str],
    data_quality: Dict[str, Any]
) -> Dict[str, Any]:
    """Build empty result structure when analysis cannot be performed."""
    return {
        "subject_analysis": {},
        "strongest_subjects": [],
        "areas_for_improvement": [],
        "overall_subject_average": None,
        "attendance": {
            "indicator": "attendance",
            "column_name": INDICATOR_COLUMNS["attendance"],
            "student_value": None,
            "class_average": None,
            "difference": None,
            "available": False,
            "data_quality": {
                "student_value_available": False,
                "class_data_available": False,
                "student_value_valid": False,
                "class_data_valid": False,
            },
        },
        "assignment_completion": {
            "indicator": "assignment_completion",
            "column_name": INDICATOR_COLUMNS["assignment_completion"],
            "student_value": None,
            "class_average": None,
            "difference": None,
            "available": False,
            "data_quality": {
                "student_value_available": False,
                "class_data_available": False,
                "student_value_valid": False,
                "class_data_valid": False,
            },
        },
        "data_quality": data_quality,
        "analysis_warnings": analysis_warnings,
    }


def to_dict_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert analysis result to a JSON-serializable dictionary.
    This ensures all numpy/pandas types are converted to native Python types.
    """
    def convert_value(val):
        if isinstance(val, dict):
            return {k: convert_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [convert_value(item) for item in val]
        elif isinstance(val, (np.integer, np.floating)):
            if isinstance(val, (np.integer,)):
                return int(val)
            else:
                float_val = float(val)
                if math.isnan(float_val) or math.isinf(float_val):
                    return None
                return float_val
        elif isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif isinstance(val, np.bool_):
            return bool(val)
        else:
            return val

    return convert_value(result)
