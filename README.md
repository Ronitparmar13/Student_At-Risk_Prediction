# Student Success & At-Risk Prediction

Streamlit dashboard for educators to review student progress, identify at-risk students, and generate AI-assisted academic guidance.

## Features
- Dataset upload and validation
- Overview analytics
- At-risk student review
- Student support emails
- AI Study Guide (Groq structured guidance)
- AI Tutor chat

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Groq configuration
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## Tests
```bash
python -m pytest tests/ -v
```
