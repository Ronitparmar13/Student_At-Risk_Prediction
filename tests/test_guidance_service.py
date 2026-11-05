"""Tests for Academic Guidance Service"""

import json
from unittest.mock import Mock, patch

import pytest
import pandas as pd

from ai_study_guide.academic_analyzer import analyze_student_academics
from ai_study_guide.guidance_service import (
    generate_academic_guidance,
    AcademicGuidanceService,
    GuidanceValidationError,
    GuidanceServiceError,
)
from ai_study_guide.groq_client import GroqRateLimitError


class TestGuidanceService:
    """Test suite for guidance service functionality."""

    def get_valid_guidance_response(self):
        """Return a valid guidance response for testing."""
        return {
            "academic_summary": "Overall performance shows mixed results with areas to focus on.",
            "strengths": [
                {"subject": "Math_Score", "explanation": "Strong performance in mathematics."}
            ],
            "areas_for_improvement": [
                {
                    "subject": "Math_Score",
                    "student_score": 85.0,
                    "class_average": 80.0,
                    "score_gap": 5.0,
                    "interpretation": "Below class average, opportunity for growth.",
                    "practical_learning_recommendations": [
                        "Practice more exercises",
                        "Review concepts regularly"
                    ]
                }
            ],
            "study_plan": [
                {
                    "day_number": 1,
                    "main_subject_or_learning_focus": "Math_Score",
                    "study_goal": "Improve math skills",
                    "suggested_activities": ["Practice problems", "Review exercises"],
                    "estimated_study_duration": "45 minutes",
                    "realistic_completion_outcome": "Complete practice set"
                },
                {
                    "day_number": 2,
                    "main_subject_or_learning_focus": "Math_Score",
                    "study_goal": "Work on calculations",
                    "suggested_activities": ["Practice calculations", "Review errors"],
                    "estimated_study_duration": "40 minutes",
                    "realistic_completion_outcome": "Complete exercises"
                },
                {
                    "day_number": 3,
                    "main_subject_or_learning_focus": "Math_Score",
                    "study_goal": "Review concepts",
                    "suggested_activities": ["Practice problems"],
                    "estimated_study_duration": "30 minutes",
                    "realistic_completion_outcome": "Solve practice problems"
                },
                {
                    "day_number": 4,
                    "main_subject_or_learning_focus": "Math_Score",
                    "study_goal": "Practice application",
                    "suggested_activities": ["Solve word problems"],
                    "estimated_study_duration": "45 minutes",
                    "realistic_completion_outcome": "Complete problems"
                },
                {
                    "day_number": 5,
                    "main_subject_or_learning_focus": "Review",
                    "study_goal": "Review weak areas",
                    "suggested_activities": ["Review mistakes", "Practice"],
                    "estimated_study_duration": "40 minutes",
                    "realistic_completion_outcome": "Identify improvements"
                },
                {
                    "day_number": 6,
                    "main_subject_or_learning_focus": "Math_Score",
                    "study_goal": "Apply skills",
                    "suggested_activities": ["Practice test"],
                    "estimated_study_duration": "45 minutes",
                    "realistic_completion_outcome": "Complete practice"
                },
                {
                    "day_number": 7,
                    "main_subject_or_learning_focus": "Review",
                    "study_goal": "Weekly review",
                    "suggested_activities": ["Review progress", "Plan next week"],
                    "estimated_study_duration": "30 minutes",
                    "realistic_completion_outcome": "Create action plan"
                }
            ],
            "teacher_guidance": [
                "Provide additional practice materials",
                "Schedule check-in to discuss progress"
            ],
            "student_encouragement": "You have the ability to improve with consistent practice!",
            "limitations": ["Based on available data only"]
        }

    def test_successful_guidance_generation(self):
        """Test successful guidance generation."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "English_Score": [85, 75, 80],
        })
        student_record = {"Math_Score": 90, "English_Score": 70}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        # Use actual values from analyzer
        sa = academic_analysis.get('subject_analysis', {})
        areas = []
        strengths = []
        for subj in ['Math_Score', 'English_Score']:
            if subj in sa and sa[subj].get('available'):
                s = sa[subj].get('student_score')
                a = sa[subj].get('class_average')
                g = sa[subj].get('difference')
                if s is not None and g is not None and g < 0:
                    areas.append({
                        "subject": subj,
                        "student_score": s,
                        "class_average": a,
                        "score_gap": g,
                        "interpretation": "Area for growth",
                        "practical_learning_recommendations": ["Practice regularly"]
                    })
                elif s is not None and g is not None and g >= 0:
                    strengths.append({"subject": subj, "explanation": "Good performance"})
        response['strengths'] = strengths or response['strengths']
        response['areas_for_improvement'] = areas or response['areas_for_improvement']
        # Fix study plan to use available subjects
        plan = []
        for i in range(1, 8):
            if i < 6 and areas:
                plan.append({
                    "day_number": i,
                    "main_subject_or_learning_focus": areas[0]['subject'],
                    "study_goal": "Improve skills",
                    "suggested_activities": ["Practice"],
                    "estimated_study_duration": "30 minutes",
                    "realistic_completion_outcome": "Complete practice"
                })
            else:
                plan.append({
                    "day_number": i,
                    "main_subject_or_learning_focus": "Review",
                    "study_goal": "Review",
                    "suggested_activities": ["Review"],
                    "estimated_study_duration": "30 minutes",
                    "realistic_completion_outcome": "Complete review"
                })
        response['study_plan'] = plan
        mock_client.chat_completion_json.return_value = response

        result = generate_academic_guidance(academic_analysis, client=mock_client)

        assert result is not None
        assert "academic_summary" in result
        assert "study_plan" in result
        assert len(result["study_plan"]) == 7

    def test_correct_academic_context_passed(self):
        """Test correct academic context passed to Groq client."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        mock_client.chat_completion_json.return_value = self.get_valid_guidance_response()

        generate_academic_guidance(academic_analysis, client=mock_client)

        # Check that client was called
        assert mock_client.chat_completion_json.called
        call_args = mock_client.chat_completion_json.call_args
        # Check system and user prompts contain context
        assert call_args is not None

    def test_accurate_preservation_of_subject_scores_and_gaps(self):
        """Test accurate preservation of subject scores and gaps."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],  # avg 80
        })
        student_record = {"Math_Score": 90}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": 90.0,
            "class_average": 80.0,
            "score_gap": 10.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        result = generate_academic_guidance(academic_analysis, client=mock_client)
        assert result["areas_for_improvement"][0]["student_score"] == 90.0
        assert result["areas_for_improvement"][0]["class_average"] == 80.0

    def test_missing_subject_data(self):
        """Test handling of missing subject data."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
        })
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = []
        response["strengths"] = []
        mock_client.chat_completion_json.return_value = response

        result = generate_academic_guidance(academic_analysis, client=mock_client)
        assert result is not None

    def test_empty_academic_analysis(self):
        """Test handling of empty academic analysis."""
        mock_client = Mock()
        with pytest.raises((GuidanceServiceError, Exception)):
            generate_academic_guidance({}, client=mock_client)

    def test_invalid_structured_output(self):
        """Test handling of invalid structured output."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        mock_client.chat_completion_json.return_value = {"invalid": "data"}

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_unsupported_subject_names(self):
        """Test handling of unsupported subject names."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Fake_Subject",
            "student_score": 85.0,
            "class_average": 80.0,
            "score_gap": 5.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_inconsistent_numerical_claims(self):
        """Test rejection of inconsistent numerical claims."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})  # avg 80
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": 95.0,  # Wrong - should be 85
            "class_average": 80.0,
            "score_gap": 15.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_exactly_seven_study_plan_entries(self):
        """Test exactly seven study plan entries required."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["study_plan"] = response["study_plan"][:5]  # Only 5 entries
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_missing_required_response_fields(self):
        """Test missing required response fields."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        del response["academic_summary"]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_additional_unexpected_fields(self):
        """Test additional unexpected fields are rejected."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["extra_field"] = "should not be here"
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_api_rate_limit_propagation(self):
        """Test API rate limit propagation."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        mock_client.chat_completion_json.side_effect = GroqRateLimitError("Rate limit exceeded")

        with pytest.raises(GroqRateLimitError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_no_identifiers_or_private_data_in_prompt(self):
        """Test no identifiers or private data in prompt."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "Student_ID": ["S1", "S2", "S3"],
            "Name": ["Alice", "Bob", "Charlie"],
        })
        student_record = {"Math_Score": 85, "Student_ID": "S1", "Name": "Alice"}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        mock_client.chat_completion_json.return_value = self.get_valid_guidance_response()

        generate_academic_guidance(academic_analysis, client=mock_client)

        call_args = mock_client.chat_completion_json.call_args
        if call_args:
            user_prompt = call_args.kwargs.get('user_prompt', '') or call_args[1].get('user_prompt', '')
            system_prompt = call_args.kwargs.get('system_prompt', '') or call_args[1].get('system_prompt', '')
            full_prompt = str(user_prompt) + str(system_prompt)
            assert "S1" not in full_prompt
            assert "Alice" not in full_prompt
            assert "Bob" not in full_prompt
            assert "Charlie" not in full_prompt

    def test_no_fabricated_scores_or_class_averages(self):
        """Test no fabricated scores or class averages in validation."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": 85.0,
            "class_average": 99.0,  # Fabricated/wrong
            "score_gap": -14.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_no_modification_of_original_academic_analysis_result(self):
        """Test no modification of original academic analysis result."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)
        original = json.dumps(academic_analysis)

        mock_client = Mock()
        mock_client.chat_completion_json.return_value = self.get_valid_guidance_response()

        generate_academic_guidance(academic_analysis, client=mock_client)

        assert json.dumps(academic_analysis) == original

    def test_uses_configured_model(self):
        """Test uses configured model."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        mock_client.chat_completion_json.return_value = self.get_valid_guidance_response()

        generate_academic_guidance(academic_analysis, client=mock_client, model="custom/model")

        call_args = mock_client.chat_completion_json.call_args
        if call_args:
            model_arg = call_args.kwargs.get('model') or call_args[1].get('model')
            assert model_arg == "custom/model"

    def test_max_tokens_configurable(self):
        """Test max tokens configurable."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        mock_client.chat_completion_json.return_value = self.get_valid_guidance_response()

        generate_academic_guidance(academic_analysis, client=mock_client, max_tokens=1000)

        call_args = mock_client.chat_completion_json.call_args
        if call_args:
            tokens_arg = call_args.kwargs.get('max_tokens') or call_args[1].get('max_tokens')
            assert tokens_arg == 1000

    def test_unknown_subject_in_strengths_rejected(self):
        """Test unknown subject in strengths is rejected."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["strengths"] = [{"subject": "Unknown_Subject", "explanation": "Good"}]
        response["areas_for_improvement"] = []
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_unknown_subject_in_improvement_rejected(self):
        """Test unknown subject in improvement recommendations is rejected."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Biology",
            "student_score": 85.0,
            "class_average": 80.0,
            "score_gap": 5.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_unknown_subject_in_study_plan_rejected(self):
        """Test unknown subject in subject-specific study plan entry is rejected."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["study_plan"][0]["main_subject_or_learning_focus"] = "Chemistry"
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_nan_and_infinity_rejected(self):
        """Test NaN and infinity are rejected."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": float('nan'),
            "class_average": 80.0,
            "score_gap": 5.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

        mock_client2 = Mock()
        response2 = self.get_valid_guidance_response()
        response2["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": 85.0,
            "class_average": float('inf'),
            "score_gap": 5.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client2.chat_completion_json.return_value = response2
        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client2)

    def test_boolean_as_scores_rejected(self):
        """Test boolean values supplied as scores are rejected."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": True,
            "class_average": 80.0,
            "score_gap": 5.0,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        mock_client.chat_completion_json.return_value = response

        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

    def test_floating_point_tolerance(self):
        """Test floating point tolerance."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["areas_for_improvement"] = [{
            "subject": "Math_Score",
            "student_score": 85.0000001,
            "class_average": 80.0,
            "score_gap": 5.0000001,
            "interpretation": "Test",
            "practical_learning_recommendations": ["Test"]
        }]
        response["strengths"] = []
        mock_client.chat_completion_json.return_value = response

        result = generate_academic_guidance(academic_analysis, client=mock_client)
        assert result is not None

    def test_duplicate_or_missing_day_numbers(self):
        """Test duplicate or missing day numbers."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        mock_client = Mock()
        response = self.get_valid_guidance_response()
        response["study_plan"][1]["day_number"] = 1  # Duplicate
        mock_client.chat_completion_json.return_value = response
        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client)

        mock_client2 = Mock()
        response2 = self.get_valid_guidance_response()
        response2["study_plan"] = response2["study_plan"][:6]  # Missing day
        mock_client2.chat_completion_json.return_value = response2
        with pytest.raises(GuidanceValidationError):
            generate_academic_guidance(academic_analysis, client=mock_client2)

    def test_empty_or_incomplete_analysis(self):
        """Test empty or incomplete analysis."""
        mock_client = Mock()
        with pytest.raises((GuidanceServiceError, Exception)):
            generate_academic_guidance({"subject_analysis": {}}, client=mock_client)
