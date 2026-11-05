"""Groq Cloud API client for Student Success & At-Risk Prediction."""

import json
import os
from typing import Any, Dict, Optional, Union

try:
    from groq import Groq
    from groq import BadRequestError, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError
    from groq import APIStatusError
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    BadRequestError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception
    APIConnectionError = Exception
    APITimeoutError = Exception
    APIStatusError = Exception
    GROQ_AVAILABLE = False


class GroqConfigurationError(Exception):
    """Raised when Groq client is misconfigured."""
    pass


class GroqAuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class GroqRateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class GroqTimeoutError(Exception):
    """Raised when request times out."""
    pass


class GroqConnectionError(Exception):
    """Raised when connection fails."""
    pass


class GroqAPIError(Exception):
    """Raised when API returns an error."""
    pass


class GroqResponseError(Exception):
    """Raised when response is malformed or unexpected."""
    pass


class GroqClient:
    """
    Reusable Groq Cloud API client.
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_MAX_TOKENS = 1500

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if not GROQ_AVAILABLE or Groq is None:
            raise GroqConfigurationError("groq package is not installed")

        self.api_key = self._load_api_key(api_key)
        self.model = self._load_model(model)
        self.timeout = timeout
        self.max_tokens = max_tokens

        try:
            self.client = Groq(api_key=self.api_key, timeout=timeout, max_retries=0)
        except Exception as e:
            self._handle_exception(e)

    def _load_api_key(self, api_key: Optional[str] = None) -> str:
        """Load API key from parameter, Streamlit secrets, or environment."""
        if api_key:
            return api_key

        for env_name in ["GROQ_API_KEY", "groq_api_key"]:
            if env_name in os.environ and os.environ[env_name]:
                return os.environ[env_name]

        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for key_name in ["GROQ_API_KEY", "groq_api_key"]:
                    try:
                        if key_name in st.secrets and st.secrets[key_name]:
                            return str(st.secrets[key_name])
                    except Exception:
                        continue
        except Exception:
            pass

        raise GroqConfigurationError("GROQ_API_KEY not found in secrets or environment")

    def _load_model(self, model: Optional[str] = None) -> str:
        """Load model from parameter, Streamlit secrets, or environment."""
        if model:
            return model

        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for key_name in ["GROQ_MODEL", "groq_model"]:
                    try:
                        if key_name in st.secrets and st.secrets[key_name]:
                            return str(st.secrets[key_name])
                    except Exception:
                        continue
        except Exception:
            pass

        for env_name in ["GROQ_MODEL", "groq_model"]:
            if env_name in os.environ and os.environ[env_name]:
                return os.environ[env_name]

        return self.DEFAULT_MODEL

    def _handle_exception(self, e: Exception) -> None:
        error_str = str(e)
        if isinstance(e, AuthenticationError) or "authentication" in error_str.lower() or "api key" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
            raise GroqAuthenticationError("Authentication failed") from e
        if isinstance(e, RateLimitError) or "rate limit" in error_str.lower() or "429" in error_str:
            raise GroqRateLimitError("Rate limit exceeded") from e
        if isinstance(e, (APITimeoutError,)) or "timeout" in error_str.lower():
            raise GroqTimeoutError("Request timed out") from e
        if isinstance(e, APIConnectionError) or "connection" in error_str.lower():
            raise GroqConnectionError("Connection failed") from e
        if isinstance(e, APIStatusError):
            status_code = getattr(e, "status_code", None)
            if status_code == 429:
                raise GroqRateLimitError("Rate limit exceeded") from e
            elif status_code in (401, 403):
                raise GroqAuthenticationError("Authentication failed") from e
            msg = getattr(e, "message", None) or str(e)
            raise GroqAPIError(f"API error (HTTP {status_code}): {msg}") from e
        if "groq" in error_str.lower() or hasattr(e, "response"):
            raise GroqAPIError("API request failed") from e
        if isinstance(e, (GroqConfigurationError, GroqAuthenticationError, GroqRateLimitError,
                          GroqTimeoutError, GroqConnectionError, GroqAPIError, GroqResponseError)):
            raise
        raise GroqAPIError("Unexpected error") from e

    def chat_completion(self, messages: list, model: Optional[str] = None, temperature: float = 0.7,
                        max_tokens: Optional[int] = None, timeout: Optional[float] = None,
                        reasoning_effort: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not messages:
            raise GroqResponseError("Messages cannot be empty")
        use_model = model or self.model
        use_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        use_timeout = timeout if timeout is not None else self.timeout
        try:
            request_kwargs = {"model": use_model, "messages": messages, "temperature": temperature,
                            "max_tokens": use_max_tokens, "timeout": use_timeout}
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            request_kwargs.update(kwargs)
            response = self.client.chat.completions.create(**request_kwargs)
            if not response:
                raise GroqResponseError("Empty response from API")
            result = {"id": getattr(response, "id", None), "model": getattr(response, "model", use_model), "choices": [], "usage": None}
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    choice_dict = {"index": getattr(choice, "index", 0), "message": {}, "finish_reason": getattr(choice, "finish_reason", None)}
                    if hasattr(choice, "message"):
                        message = choice.message
                        choice_dict["message"] = {"role": getattr(message, "role", None), "content": getattr(message, "content", None)}
                    result["choices"].append(choice_dict)
            else:
                raise GroqResponseError("Response contains no choices")
            if hasattr(response, "usage"):
                usage = response.usage
                result["usage"] = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                                   "completion_tokens": getattr(usage, "completion_tokens", None),
                                   "total_tokens": getattr(usage, "total_tokens", None)}
            return result
        except Exception as e:
            if isinstance(e, (GroqConfigurationError, GroqAuthenticationError, GroqRateLimitError,
                              GroqTimeoutError, GroqConnectionError, GroqAPIError, GroqResponseError)):
                raise
            self._handle_exception(e)

    def chat_completion_text(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            timeout: Optional[float] = None, reasoning_effort: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.chat_completion(messages=messages, model=model, temperature=temperature,
                                       max_tokens=max_tokens, timeout=timeout, reasoning_effort=reasoning_effort)
        try:
            content = response["choices"][0]["message"]["content"]
            if content is None:
                raise GroqResponseError("Response content is None")
            return content
        except (KeyError, IndexError) as e:
            raise GroqResponseError("Malformed response structure") from e

    def chat_completion_json(self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any],
                            schema_name: str = "response", model: Optional[str] = None,
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            timeout: Optional[float] = None, reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
        if not json_schema:
            raise GroqResponseError("JSON schema cannot be empty")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response_format = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": json_schema, "strict": True}}
            response = self.chat_completion(messages=messages, model=model, temperature=temperature,
                                           max_tokens=max_tokens, timeout=timeout,
                                           reasoning_effort=reasoning_effort, response_format=response_format)
            try:
                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise GroqResponseError("Response content is None")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise GroqResponseError("Response is not a JSON object")
                return parsed
            except json.JSONDecodeError as e:
                raise GroqResponseError("Malformed JSON response") from e
            except (KeyError, IndexError) as e:
                raise GroqResponseError("Malformed response structure") from e
        except Exception as e:
            if isinstance(e, (GroqConfigurationError, GroqAuthenticationError, GroqRateLimitError,
                              GroqTimeoutError, GroqConnectionError, GroqAPIError, GroqResponseError)):
                raise
            self._handle_exception(e)

    def get_model(self) -> str:
        return self.model

    def set_model(self, model: str) -> None:
        if not model:
            raise GroqConfigurationError("Model cannot be empty")
        self.model = model
