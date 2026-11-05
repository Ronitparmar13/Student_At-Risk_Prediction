"""Integration tests for AI Study Guide: Analyzer -> Guidance Service -> Groq Client"""

import json
from unittest.mock import Mock, patch

import pytest
import pandas as pd

from ai_study_guide.academic_analyzer import analyze_student_academics
from ai_study_guide.groq_client import (
    GroqClient,
    GroqAuthenticationError,
    GroqRateLimitError,
)
from ai_study_guide.guidance_service import (
    generate_academic_guidance,
    AcademicGuidanceService,
    GuidanceValidationError,
)


class TestAIStudyGuideIntegration:
    """Integration tests for the complete AI Study Guide workflow."""

    def test_successful_analysis_and_guidance_generation(self):
        """Test complete workflow: analyzer -> guidance service -> groq client."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70, 85],
            "English_Score": [75, 95, 80, 90],
        })
        student_record = {"Math_Score": 95, "English_Score": 70}
        academic_analysis = analyze_student_academics(student_record, class_df)

        sa = academic_analysis["subject_analysis"]
        assert sa["Math_Score"]["available"] is True
        assert sa["English_Score"]["available"] is True

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            # Build valid response using actual analyzer values
            strengths = []
            areas = []
            for subj in ["Math_Score", "English_Score"]:
                if subj in sa and sa[subj]["available"]:
                    s = sa[subj]["student_score"]
                    a = sa[subj]["class_average"]
                    g = sa[subj]["difference"]
                    if g < 0:
                        areas.append({
                            "subject": subj,
                            "student_score": s,
                            "class_average": a,
                            "score_gap": g,
                            "interpretation": "Opportunity for growth",
                            "practical_learning_recommendations": ["Practice regularly", "Seek help"]
                        })
                    else:
                        strengths.append({"subject": subj, "explanation": "Strong performance"})

            plan = []
            for i in range(1, 8):
                if areas and i <= 5:
                    plan.append({
                        "day_number": i,
                        "main_subject_or_learning_focus": areas[0]["subject"],
                        "study_goal": "Improve skills",
                        "suggested_activities": ["Practice exercises"],
                        "estimated_study_duration": "45 minutes",
                        "realistic_completion_outcome": "Complete practice set"
                    })
                elif strengths and i <= 3:
                    plan.append({
                        "day_number": i,
                        "main_subject_or_learning_focus": strengths[0]["subject"],
                        "study_goal": "Maintain skills",
                        "suggested_activities": ["Review concepts"],
                        "estimated_study_duration": "30 minutes",
                        "realistic_completion_outcome": "Review complete"
                    })
                else:
                    plan.append({
                        "day_number": i,
                        "main_subject_or_learning_focus": "Review",
                        "study_goal": "Weekly review",
                        "suggested_activities": ["Review progress"],
                        "estimated_study_duration": "30 minutes",
                        "realistic_completion_outcome": "Plan next steps"
                    })

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Mixed performance with clear areas for improvement.",
                "strengths": strengths,
                "areas_for_improvement": areas,
                "study_plan": plan,
                "teacher_guidance": ["Provide targeted support", "Regular check-ins"],
                "student_encouragement": "Keep up the good work and focus on growth areas.",
                "limitations": ["Based on available assessment data"]
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            result = generate_academic_guidance(academic_analysis, client=client)

            assert result is not None
            assert "academic_summary" in result
            assert len(result["study_plan"]) == 7
            assert mock_client_instance.chat.completions.create.called

    def test_exact_preservation_of_scores_averages_gaps(self):
        """Test exact preservation of subject scores and gaps."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "English_Score": [60, 80, 100],
        })
        student_record = {"Math_Score": 90, "English_Score": 70}
        academic_analysis = analyze_student_academics(student_record, class_df)

        sa = academic_analysis["subject_analysis"]

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            areas = []
            for subj in ["Math_Score", "English_Score"]:
                if subj in sa and sa[subj]["available"]:
                    areas.append({
                        "subject": subj,
                        "student_score": sa[subj]["student_score"],
                        "class_average": sa[subj]["class_average"],
                        "score_gap": sa[subj]["difference"],
                        "interpretation": "Growth opportunity",
                        "practical_learning_recommendations": ["Practice"]
                    })

            plan = []
            for i in range(1, 8):
                plan.append({
                    "day_number": i,
                    "main_subject_or_learning_focus": "Review" if i > 4 else areas[0]["subject"],
                    "study_goal": "Study",
                    "suggested_activities": ["Practice"],
                    "estimated_study_duration": "30 minutes",
                    "realistic_completion_outcome": "Done"
                })

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Summary",
                "strengths": [],
                "areas_for_improvement": areas,
                "study_plan": plan,
                "teacher_guidance": ["Support"],
                "student_encouragement": "Keep going",
                "limitations": []
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            result = generate_academic_guidance(academic_analysis, client=client)

            for area in result["areas_for_improvement"]:
                subj = area["subject"]
                assert abs(area["student_score"] - sa[subj]["student_score"]) < 1e-6
                assert abs(area["class_average"] - sa[subj]["class_average"]) < 1e-6
                assert abs(area["score_gap"] - sa[subj]["difference"]) < 1e-6

    def test_strict_rejection_of_fabricated_numerical_values(self):
        """Test strict rejection of fabricated numerical values."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Summary",
                "strengths": [],
                "areas_for_improvement": [{
                    "subject": "Math_Score",
                    "student_score": 95.0,  # Fabricated
                    "class_average": 80.0,
                    "score_gap": 15.0,
                    "interpretation": "Test",
                    "practical_learning_recommendations": ["Practice"]
                }],
                "study_plan": [{"day_number": i, "main_subject_or_learning_focus": "Math_Score", "study_goal": "g", "suggested_activities": ["a"], "estimated_study_duration": "30m", "realistic_completion_outcome": "ok"} for i in range(1,8)],
                "teacher_guidance": ["Support"],
                "student_encouragement": "Go",
                "limitations": []
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            with pytest.raises(GuidanceValidationError):
                generate_academic_guidance(academic_analysis, client=client)

    def test_strict_rejection_of_unsupported_subject(self):
        """Test strict rejection of unsupported subject."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Summary",
                "strengths": [{"subject": "Biology", "explanation": "Good"}],
                "areas_for_improvement": [],
                "study_plan": [{"day_number": i, "main_subject_or_learning_focus": "Review", "study_goal": "g", "suggested_activities": ["a"], "estimated_study_duration": "30m", "realistic_completion_outcome": "ok"} for i in range(1,8)],
                "teacher_guidance": ["Support"],
                "student_encouragement": "Go",
                "limitations": []
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            with pytest.raises(GuidanceValidationError):
                generate_academic_guidance(academic_analysis, client=client)

    def test_handling_of_missing_unavailable_subject_data(self):
        """Test handling of missing or unavailable subject data."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            sa = academic_analysis["subject_analysis"]
            strengths = []
            if "Math_Score" in sa and sa["Math_Score"]["available"]:
                strengths.append({"subject": "Math_Score", "explanation": "Good"})

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Summary",
                "strengths": strengths,
                "areas_for_improvement": [],
                "study_plan": [{"day_number": i, "main_subject_or_learning_focus": "Review", "study_goal": "g", "suggested_activities": ["a"], "estimated_study_duration": "30m", "realistic_completion_outcome": "ok"} for i in range(1,8)],
                "teacher_guidance": ["Support"],
                "student_encouragement": "Go",
                "limitations": ["Some subjects have insufficient data"]
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            result = generate_academic_guidance(academic_analysis, client=client)
            assert result is not None

    def test_groq_authentication_failure_propagation(self):
        """Test Groq authentication failure propagation."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance
            auth_err = Exception("Invalid API key")
            auth_err.status_code = 401
            mock_client_instance.chat.completions.create.side_effect = auth_err

            client = GroqClient(api_key="test-key")
            with pytest.raises(GroqAuthenticationError):
                generate_academic_guidance(academic_analysis, client=client)

    def test_groq_rate_limit_propagation_without_retries(self):
        """Test Groq rate limit propagation without retries."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance
            rate_err = Exception("Rate limit exceeded")
            rate_err.status_code = 429
            mock_client_instance.chat.completions.create.side_effect = rate_err

            client = GroqClient(api_key="test-key")
            with pytest.raises(GroqRateLimitError):
                generate_academic_guidance(academic_analysis, client=client)
            assert mock_client_instance.chat.completions.create.call_count == 1

    def test_malformed_or_invalid_structured_responses(self):
        """Test malformed or invalid structured responses."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "not json"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            with pytest.raises(Exception):
                generate_academic_guidance(academic_analysis, client=client)

    def test_privacy_no_identifiers_in_prompt(self):
        """Test that student identifiers and personal data don't reach Groq prompt."""
        class_df = pd.DataFrame({
            "Math_Score": [80, 90, 70],
            "Student_ID": ["S1", "S2", "S3"],
            "Name": ["Alice", "Bob", "Charlie"],
        })
        student_record = {"Math_Score": 85, "Student_ID": "S1", "Name": "Alice"}
        academic_analysis = analyze_student_academics(student_record, class_df)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            sa = academic_analysis["subject_analysis"]
            strengths = [{"subject": "Math_Score", "explanation": "Good"}] if sa.get("Math_Score", {}).get("available") else []

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Summary",
                "strengths": strengths,
                "areas_for_improvement": [],
                "study_plan": [{"day_number": i, "main_subject_or_learning_focus": "Review", "study_goal": "g", "suggested_activities": ["a"], "estimated_study_duration": "30m", "realistic_completion_outcome": "ok"} for i in range(1,8)],
                "teacher_guidance": ["Support"],
                "student_encouragement": "Go",
                "limitations": []
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            result = generate_academic_guidance(academic_analysis, client=client)

            call_args = mock_client_instance.chat.completions.create.call_args
            if call_args:
                kwargs = call_args.kwargs
                messages = kwargs.get('messages', [])
                full_text = json.dumps(messages)
                assert "S1" not in full_text
                assert "Alice" not in full_text
                assert "Bob" not in full_text
                assert "Charlie" not in full_text

    def test_original_analyzer_data_unchanged(self):
        """Test original analyzer data remains unchanged."""
        class_df = pd.DataFrame({"Math_Score": [80, 90, 70]})
        student_record = {"Math_Score": 85}
        academic_analysis = analyze_student_academics(student_record, class_df)
        original = json.dumps(academic_analysis)

        with patch('ai_study_guide.groq_client.Groq') as mock_groq:
            mock_client_instance = Mock()
            mock_groq.return_value = mock_client_instance

            sa = academic_analysis["subject_analysis"]
            strengths = [{"subject": "Math_Score", "explanation": "Good"}] if sa.get("Math_Score", {}).get("available") else []

            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = json.dumps({
                "academic_summary": "Summary",
                "strengths": strengths,
                "areas_for_improvement": [],
                "study_plan": [{"day_number": i, "main_subject_or_learning_focus": "Review", "study_goal": "g", "suggested_activities": ["a"], "estimated_study_duration": "30m", "realistic_completion_outcome": "ok"} for i in range(1,8)],
                "teacher_guidance": ["Support"],
                "student_encouragement": "Go",
                "limitations": []
            })
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client_instance.chat.completions.create.return_value = mock_response

            client = GroqClient(api_key="test-key")
            generate_academic_guidance(academic_analysis, client=client)

            assert json.dumps(academic_analysis) == original
