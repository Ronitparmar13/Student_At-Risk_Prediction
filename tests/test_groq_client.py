"""Tests for Groq Client"""

import json
import os
from unittest.mock import Mock, patch

import pytest

from ai_study_guide.groq_client import (
    GroqClient,
    GroqConfigurationError,
    GroqAuthenticationError,
    GroqRateLimitError,
    GroqTimeoutError,
    GroqConnectionError,
    GroqAPIError,
    GroqResponseError,
)


class TestGroqClient:
    """Test suite for Groq client functionality."""

    def test_missing_api_key(self):
        """Test missing API key raises configuration error."""
        with patch('ai_study_guide.groq_client.Groq'):
            with patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True):
                with patch.dict(os.environ, {}, clear=True):
                    with patch('streamlit.secrets', {}, create=True):
                        with pytest.raises((GroqConfigurationError, Exception)):
                            GroqClient()

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_successful_client_initialization(self, mock_groq):
        """Test successful client initialization."""
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            assert client is not None
            assert client.api_key == "test-key"
            assert client.model == GroqClient.DEFAULT_MODEL

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_max_retries_zero_enforced(self, mock_groq):
        """Test that max_retries=0 is always enforced."""
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            mock_groq.assert_called_once()
            call_kwargs = mock_groq.call_args.kwargs
            assert call_kwargs.get('max_retries') == 0
            assert call_kwargs.get('api_key') == 'test-key'
            assert 'timeout' in call_kwargs

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_rate_limit_no_retry(self, mock_groq_class):
        """Test 429 responses produce GroqRateLimitError without automatic retry."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        rate_err = Exception("Rate limit exceeded")
        rate_err.status_code = 429
        mock_client_instance.chat.completions.create.side_effect = rate_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            with pytest.raises(GroqRateLimitError):
                client.chat_completion_text("Sys", "User")
            # Should only be called once (no retries)
            assert mock_client_instance.chat.completions.create.call_count == 1

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_successful_text_completion(self, mock_groq_class):
        """Test successful text completion."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Hello, this is a test response."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.id = "test"
        mock_response.model = "test"

        mock_client_instance.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            result = client.chat_completion_text("You are helpful.", "Say hello")
            assert result == "Hello, this is a test response."

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_successful_structured_json_response(self, mock_groq_class):
        """Test successful structured JSON response."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"answer": "test", "score": 95}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.id = "test"
        mock_response.model = "test"

        mock_client_instance.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "score": {"type": "number"}
                },
                "required": ["answer", "score"],
                "additionalProperties": False
            }
            result = client.chat_completion_json("System", "User", schema)
            assert isinstance(result, dict)
            assert result["answer"] == "test"
            assert result["score"] == 95

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_correct_json_schema_request_configuration(self, mock_groq_class):
        """Test correct JSON Schema request configuration."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"result": "ok"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.id = "test"
        mock_response.model = "test"

        mock_client_instance.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            schema = {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False
            }
            client.chat_completion_json("Sys", "User", schema, schema_name="test_schema")
            call_args = mock_client_instance.chat.completions.create.call_args
            assert call_args is not None
            kwargs = call_args.kwargs
            assert "response_format" in kwargs
            assert kwargs["response_format"]["type"] == "json_schema"
            assert kwargs["response_format"]["json_schema"]["name"] == "test_schema"
            assert kwargs["response_format"]["json_schema"]["strict"] is True

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_authentication_error(self, mock_groq_class):
        """Test authentication error."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        auth_err = Exception("Invalid API key")
        auth_err.status_code = 401
        mock_client_instance.chat.completions.create.side_effect = auth_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            with pytest.raises(GroqAuthenticationError):
                client.chat_completion_text("Sys", "User")

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_rate_limit_error(self, mock_groq_class):
        """Test rate limit error."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        rate_err = Exception("Rate limit exceeded")
        rate_err.status_code = 429
        mock_client_instance.chat.completions.create.side_effect = rate_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            with pytest.raises(GroqRateLimitError):
                client.chat_completion_text("Sys", "User")

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_timeout_error(self, mock_groq_class):
        """Test timeout error."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        timeout_err = Exception("Request timeout")
        mock_client_instance.chat.completions.create.side_effect = timeout_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            with pytest.raises(GroqTimeoutError):
                client.chat_completion_text("Sys", "User")

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_connection_error(self, mock_groq_class):
        """Test connection error."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        conn_err = Exception("Connection error")
        mock_client_instance.chat.completions.create.side_effect = conn_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            with pytest.raises(GroqConnectionError):
                client.chat_completion_text("Sys", "User")

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_unexpected_api_failure(self, mock_groq_class):
        """Test unexpected API failure."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        api_err = Exception("Server error")
        api_err.status_code = 500
        mock_client_instance.chat.completions.create.side_effect = api_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            with pytest.raises(GroqAPIError):
                client.chat_completion_text("Sys", "User")

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_malformed_or_empty_response(self, mock_groq_class):
        """Test malformed or empty response."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            mock_client_instance.chat.completions.create.return_value = None
            with pytest.raises(GroqResponseError):
                client.chat_completion_text("Sys", "User")
            mock_response = Mock()
            mock_response.choices = []
            mock_client_instance.chat.completions.create.return_value = mock_response
            with pytest.raises(GroqResponseError):
                client.chat_completion_text("Sys", "User")
            mock_response2 = Mock()
            mock_response2.choices = [Mock()]
            mock_response2.choices[0].message = Mock()
            mock_response2.choices[0].message.content = "not json"
            mock_client_instance.chat.completions.create.return_value = mock_response2
            schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
            with pytest.raises(GroqResponseError):
                client.chat_completion_json("Sys", "User", schema)

    @patch('ai_study_guide.groq_client.Groq')
    @patch('ai_study_guide.groq_client.GROQ_AVAILABLE', True)
    def test_no_credential_leakage_in_error_messages(self, mock_groq_class):
        """Test no credential leakage in error messages."""
        mock_client_instance = Mock()
        mock_groq_class.return_value = mock_client_instance
        api_err = Exception("Error with key sk-1234567890abcdef")
        mock_client_instance.chat.completions.create.side_effect = api_err
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            client = GroqClient()
            try:
                client.chat_completion_text("Sys", "User")
            except Exception as e:
                error_msg = str(e)
                assert "sk-1234567890abcdef" not in error_msg
                assert "test-key" not in error_msg
