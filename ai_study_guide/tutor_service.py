"""AI Tutor service for Student Success & At-Risk Prediction."""

from typing import Any, Dict, List, Optional

from ai_study_guide.groq_client import GroqClient, GroqAPIError, GroqAuthenticationError, GroqRateLimitError, GroqTimeoutError, GroqConnectionError, GroqConfigurationError, GroqResponseError

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY = 20


class TutorService:
    def __init__(self, client: Optional[GroqClient] = None):
        self.client = client

    def _ensure_client(self):
        if self.client is None:
            self.client = GroqClient()
        return self.client

    def _build_context(self, academic_context: Optional[Dict[str, Any]] = None) -> str:
        if not academic_context:
            return ""
        parts = []
        subjects = academic_context.get("subjects", [])
        if subjects:
            parts.append("Available subjects: " + ", ".join(subjects))
        scores = academic_context.get("scores", {})
        for subj, info in scores.items():
            if isinstance(info, dict) and info.get("available"):
                parts.append(f"{subj}: student={info.get('student_score')}, class_avg={info.get('class_average')}, gap={info.get('difference')}")
        recs = academic_context.get("recommendations", [])
        if recs:
            parts.append("Study guide recommendations: " + "; ".join(str(r) for r in recs))
        att = academic_context.get("attendance")
        if isinstance(att, dict) and att.get("available"):
            parts.append(f"Attendance: {att.get('student_value')}% (class avg {att.get('class_average')}%)")
        assign = academic_context.get("assignment_completion")
        if isinstance(assign, dict) and assign.get("available"):
            parts.append(f"Assignment completion: {assign.get('student_value')}% (class avg {assign.get('class_average')}%)")
        return "\n".join(parts)

    def chat(self, question: str, history: Optional[List[Dict[str, str]]] = None,
             subject: Optional[str] = None, academic_context: Optional[Dict[str, Any]] = None,
             model: Optional[str] = None, max_tokens: int = 800) -> str:
        if not question or not question.strip():
            raise ValueError("Message cannot be empty")
        if len(question) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH}")
        client = self._ensure_client()
        system_prompt = ("You are a supportive AI tutor for students. Explain concepts simply, give examples, "
                         "break problems into steps, and encourage learning. Do not invent syllabus details or scores. "
                         "Distinguish general knowledge from verified academic data.")
        context = self._build_context(academic_context)
        if subject:
            context = (context + "\n" if context else "") + f"Current subject focus: {subject}"
        if context:
            system_prompt += "\n\nVerified academic context (use only these facts):\n" + context
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or [])[-MAX_HISTORY:]:
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})
        try:
            resp = client.chat_completion(messages=messages, model=model, temperature=0.7, max_tokens=max_tokens)
            content = resp["choices"][0]["message"]["content"]
            if content is None:
                raise GroqResponseError("Response content is None")
            return content
        except (GroqAuthenticationError, GroqConfigurationError, GroqRateLimitError, GroqTimeoutError, GroqConnectionError, GroqAPIError, GroqResponseError):
            raise


def create_tutor_service(client=None):
    return TutorService(client=client)
