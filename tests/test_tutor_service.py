import pytest
from unittest.mock import patch, Mock
from ai_study_guide.tutor_service import TutorService, MAX_MESSAGE_LENGTH

def test_basic():
    with patch('streamlit.secrets', {'GROQ_API_KEY': 'test-key'}, create=True):
        with patch('ai_study_guide.groq_client.Groq') as m:
            inst = Mock()
            m.return_value = inst
            resp = Mock()
            choice = Mock(); msg = Mock(); msg.content = "hi"; choice.message = msg; resp.choices=[choice]
            inst.chat.completions.create.return_value = resp
            svc = TutorService()
            out = svc.chat("hello")
            assert out == "hi"

def test_empty():
    svc = TutorService.__new__(TutorService)
    svc.client = None
    with patch('streamlit.secrets', {'GROQ_API_KEY': 'test-key'}, create=True):
        with pytest.raises(ValueError):
            svc.chat("")

def test_max_length():
    svc = TutorService.__new__(TutorService)
    svc.client = None
    with patch('streamlit.secrets', {'GROQ_API_KEY': 'test-key'}, create=True):
        with pytest.raises(ValueError):
            svc.chat("a"*(MAX_MESSAGE_LENGTH+1))
