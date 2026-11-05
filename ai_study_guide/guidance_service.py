"""AI Guidance Service for Student Success & At-Risk Prediction."""

import json
import math
from typing import Any, Dict, Optional

from ai_study_guide.academic_analyzer import analyze_student_academics
from ai_study_guide.groq_client import (
    GroqClient,
    GroqAPIError,
    GroqAuthenticationError,
    GroqConfigurationError,
    GroqConnectionError,
    GroqRateLimitError,
    GroqResponseError,
    GroqTimeoutError,
)


class GuidanceValidationError(Exception):
    pass


class GuidanceServiceError(Exception):
    pass


class AcademicGuidanceService:
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_MAX_TOKENS = 4096
    FLOAT_TOLERANCE = 1e-6

    GUIDANCE_SCHEMA = {
        "type": "object",
        "properties": {
            "academic_summary": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "object", "properties": {"subject": {"type": "string"}, "explanation": {"type": "string"}}, "required": ["subject", "explanation"], "additionalProperties": False}},
            "areas_for_improvement": {"type": "array", "items": {"type": "object", "properties": {"subject": {"type": "string"}, "student_score": {"type": "number"}, "class_average": {"type": "number"}, "score_gap": {"type": "number"}, "interpretation": {"type": "string"}, "practical_learning_recommendations": {"type": "array", "items": {"type": "string"}}}, "required": ["subject", "student_score", "class_average", "score_gap", "interpretation", "practical_learning_recommendations"], "additionalProperties": False}},
            "study_plan": {"type": "array", "items": {"type": "object", "properties": {"day_number": {"type": "integer"}, "main_subject_or_learning_focus": {"type": "string"}, "study_goal": {"type": "string"}, "suggested_activities": {"type": "array", "items": {"type": "string"}}, "estimated_study_duration": {"type": "string"}, "realistic_completion_outcome": {"type": "string"}}, "required": ["day_number", "main_subject_or_learning_focus", "study_goal", "suggested_activities", "estimated_study_duration", "realistic_completion_outcome"], "additionalProperties": False}},
            "teacher_guidance": {"type": "array", "items": {"type": "string"}},
            "student_encouragement": {"type": "string"},
            "limitations": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["academic_summary", "strengths", "areas_for_improvement", "study_plan", "teacher_guidance", "student_encouragement", "limitations"],
        "additionalProperties": False
    }

    def __init__(self, client: Optional[GroqClient] = None):
        self.client = client

    def _derive_whitelist(self, academic_analysis: Dict[str, Any]) -> set:
        whitelist = set()
        subject_analysis = academic_analysis.get("subject_analysis", {})
        if isinstance(subject_analysis, dict):
            for subject, analysis in subject_analysis.items():
                if isinstance(analysis, dict) and analysis.get("available") is True:
                    whitelist.add(subject)
        return whitelist

    def _is_valid_number(self, val: Any) -> bool:
        if isinstance(val, bool):
            return False
        if isinstance(val, int) and not isinstance(val, bool):
            return True
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return False
            return True
        try:
            import numpy as np
            if isinstance(val, (np.integer,)) and not isinstance(val, (bool, np.bool_)):
                return True
            if isinstance(val, (np.floating,)):
                if math.isnan(float(val)) or math.isinf(float(val)):
                    return False
                return True
            if isinstance(val, (np.bool_,)):
                return False
        except Exception:
            pass
        return False

    def _prepare_context(self, academic_analysis: Dict[str, Any]) -> str:
        context_parts = []
        if academic_analysis.get("overall_subject_average") is not None:
            context_parts.append(f"Overall subject average: {academic_analysis['overall_subject_average']}")
        subject_analysis = academic_analysis.get("subject_analysis", {})
        if isinstance(subject_analysis, dict):
            for subject, analysis in subject_analysis.items():
                if isinstance(analysis, dict) and analysis.get("available") and analysis.get("student_score") is not None:
                    context_parts.append(f"{subject}: student={analysis['student_score']}, class_avg={analysis['class_average']}, gap={analysis['difference']}, status={analysis['performance_status']}")
        areas = academic_analysis.get("areas_for_improvement", [])
        if areas and isinstance(areas, list):
            context_parts.append("\nPriority areas for improvement:")
            for area in areas:
                if isinstance(area, dict):
                    context_parts.append(f"- {area.get('subject')}: score {area.get('student_score')}, gap {area.get('difference')}")
        attendance = academic_analysis.get("attendance", {})
        if isinstance(attendance, dict) and attendance.get("available"):
            context_parts.append(f"Attendance: {attendance['student_value']}% (class avg {attendance['class_average']}%)")
        assignment = academic_analysis.get("assignment_completion", {})
        if isinstance(assignment, dict) and assignment.get("available"):
            context_parts.append(f"Assignment completion: {assignment['student_value']}% (class avg {assignment['class_average']}%)")
        data_quality = academic_analysis.get("data_quality", {})
        if isinstance(data_quality, dict) and data_quality.get("subjects_missing"):
            context_parts.append(f"Missing subject data: {', '.join(data_quality['subjects_missing'])}")
        warnings = academic_analysis.get("analysis_warnings", [])
        if isinstance(warnings, list) and warnings:
            context_parts.append(f"Data notes: {len(warnings)} issue(s) noted")
        return "\n".join(context_parts)

    def _validate_response(self, response: Dict[str, Any], academic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict): raise GuidanceValidationError("Response must be a dictionary")
        required_fields = ["academic_summary", "strengths", "areas_for_improvement", "study_plan", "teacher_guidance", "student_encouragement", "limitations"]
        for field in required_fields:
            if field not in response: raise GuidanceValidationError(f"Missing required field: {field}")
        for key in response.keys():
            if key not in required_fields: raise GuidanceValidationError(f"Unexpected field in response: {key}")
        study_plan = response.get("study_plan", [])
        if not isinstance(study_plan, list) or len(study_plan) != 7: raise GuidanceValidationError("Study plan must have exactly 7 entries")
        day_numbers = []
        for entry in study_plan:
            if isinstance(entry, dict) and 'day_number' in entry:
                dn = entry['day_number']
                if isinstance(dn, bool) or not isinstance(dn, int): raise GuidanceValidationError("day_number must be integer")
                day_numbers.append(dn)
        if len(set(day_numbers)) != len(day_numbers) or sorted(day_numbers) != list(range(1,8)): raise GuidanceValidationError("Day numbers must be unique and run from 1 through 7")
        whitelist = self._derive_whitelist(academic_analysis)
        subject_analysis = academic_analysis.get("subject_analysis", {})
        for strength in response.get("strengths", []):
            if isinstance(strength, dict) and strength.get("subject") not in whitelist: raise GuidanceValidationError("Unsupported or invented subject")
        for area in response.get("areas_for_improvement", []):
            if isinstance(area, dict):
                if area.get("subject") not in whitelist: raise GuidanceValidationError("Unsupported or invented subject")
                for num_field in ["student_score", "class_average", "score_gap"]:
                    if not self._is_valid_number(area.get(num_field)): raise GuidanceValidationError("Invalid numeric value")
        for entry in study_plan:
            if isinstance(entry, dict):
                focus = entry.get("main_subject_or_learning_focus")
                if isinstance(focus, str):
                    fl = focus.lower().strip()
                    is_generic = any(fl == t or fl.startswith(t) or t in fl for t in ["review", "revision", "time management", "practice", "general", "overview", "planning", "weekly", "mistake"])
                    if focus in subject_analysis and isinstance(subject_analysis[focus], dict):
                        if subject_analysis[focus].get("available"):
                            if focus not in whitelist: raise GuidanceValidationError("Unsupported or invented subject in study plan")
                        elif not is_generic: raise GuidanceValidationError("Unsupported or invented subject in study plan")
                    elif focus in whitelist:
                        if isinstance(subject_analysis.get(focus), dict) and subject_analysis[focus].get("available") is False: raise GuidanceValidationError("Unsupported or invented subject in study plan")
                    elif not is_generic:
                        if isinstance(subject_analysis, dict) and focus in subject_analysis:
                            if isinstance(subject_analysis[focus], dict):
                                if subject_analysis[focus].get("available") and focus not in whitelist: raise GuidanceValidationError("Unsupported or invented subject in study plan")
                                if subject_analysis[focus].get("available") is False: raise GuidanceValidationError("Unsupported or invented subject in study plan")
                        else: raise GuidanceValidationError("Unsupported or invented subject in study plan")
        for area in response.get("areas_for_improvement", []):
            if isinstance(area, dict) and area.get("subject") in subject_analysis and isinstance(subject_analysis[area["subject"]], dict) and subject_analysis[area["subject"]].get("available"):
                expected_score = subject_analysis[area["subject"]].get("student_score")
                expected_avg = subject_analysis[area["subject"]].get("class_average")
                expected_gap = subject_analysis[area["subject"]].get("difference")
                for num_field in ["student_score", "class_average", "score_gap"]:
                    if not self._is_valid_number(area.get(num_field)): raise GuidanceValidationError("Inconsistent numerical values")
                if expected_score is not None:
                    try:
                        if not math.isclose(float(area.get("student_score")), float(expected_score), abs_tol=self.FLOAT_TOLERANCE): raise GuidanceValidationError("Inconsistent student score")
                    except GuidanceValidationError:
                        raise
                    except Exception: raise GuidanceValidationError("Inconsistent student score")
                if expected_avg is not None:
                    try:
                        if not math.isclose(float(area.get("class_average")), float(expected_avg), abs_tol=self.FLOAT_TOLERANCE): raise GuidanceValidationError("Inconsistent class average")
                    except Exception: raise GuidanceValidationError("Inconsistent class average")
                if expected_gap is not None:
                    try:
                        if not math.isclose(float(area.get("score_gap")), float(expected_gap), abs_tol=self.FLOAT_TOLERANCE): raise GuidanceValidationError("Inconsistent score gap")
                    except Exception: raise GuidanceValidationError("Inconsistent score gap")
        return response

    def generate_academic_guidance(self, academic_analysis: Dict[str, Any], client: Optional[GroqClient] = None, model: Optional[str] = None, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.7) -> Dict[str, Any]:
        if not academic_analysis: raise GuidanceServiceError("Academic analysis cannot be empty")
        use_client = client or self.client
        if use_client is None:
            try: use_client = GroqClient()
            except Exception as e: raise GuidanceServiceError(f"Failed to initialize Groq client: {str(e)}") from e
        use_model = model or self.DEFAULT_MODEL
        context = self._prepare_context(academic_analysis)
        system_prompt = """You are a supportive academic tutor helping students improve their learning.\n\nRules:\n- Be supportive, constructive, and encouraging. Never stigmatize or say a student is unsuccessful/incapable.\n- Use ONLY the verified academic data provided. Never invent marks, class averages, subjects, or academic history.\n- Never change risk classifications or make unsupported diagnoses. Do not infer specific cognitive or subject skills from a numerical score alone; recommend teacher review of actual assignments or assessments to identify specific gaps. Avoid unsupported claims about motivation or personal characteristics.\n- Focus on achievable improvements with practical study methods.\n- If a subject has missing/unavailable data, acknowledge you cannot assess it.\n- Prioritize areas identified as needing improvement.\n- Give constructive teacher guidance.\n- Encourage help-seeking when appropriate.\n- Use the actual scores and gaps provided - do not recalculate them. Distinguish measured facts from suggested areas for investigation. Do not diagnose specific skills from aggregate scores; present topics as possible revision examples.\n- Do not claim knowledge of specific chapters/syllabus unless provided.\n- Do not make deterministic negative labels like "weak" or "failed"."""
        user_prompt = f"Based on the following verified academic analysis, generate personalized guidance.\n\n{context}\n\nCreate a complete guidance response following the required structure. The study_plan must contain exactly 7 entries with day_number values 1 through 7. Use the exact subject identifiers provided (such as Math_Score), never shortened or renamed forms. Use only the subjects and numbers provided above. Do not invent any scores or averages."
        try:
            response = use_client.chat_completion_json(system_prompt=system_prompt, user_prompt=user_prompt, json_schema=self.GUIDANCE_SCHEMA, schema_name="academic_guidance", model=use_model, temperature=temperature, max_tokens=max_tokens)
            validated = self._validate_response(response, academic_analysis)
            return validated
        except (GroqAuthenticationError, GroqConfigurationError): raise
        except GroqRateLimitError: raise
        except (GroqTimeoutError, GroqConnectionError, GroqAPIError) as e: raise GuidanceServiceError(f"API error: {str(e)}") from e
        except GroqResponseError as e: raise GuidanceValidationError(f"Invalid response from API: {str(e)}") from e
        except GuidanceValidationError: raise
        except Exception as e: raise GuidanceServiceError(f"Unexpected error: {str(e)}") from e


def generate_academic_guidance(academic_analysis: Dict[str, Any], client: Optional[GroqClient] = None, model: Optional[str] = None, max_tokens: int = AcademicGuidanceService.DEFAULT_MAX_TOKENS, temperature: float = 0.7) -> Dict[str, Any]:
    service = AcademicGuidanceService(client=client)
    return service.generate_academic_guidance(academic_analysis=academic_analysis, model=model, max_tokens=max_tokens, temperature=temperature)
