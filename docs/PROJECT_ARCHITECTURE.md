# Project Architecture

```mermaid
flowchart TD
  A[Streamlit app.py] --> B[Upload/Validation]
  A --> C[Overview]
  A --> D[At-risk students]
  A --> E[Student Support]
  A --> F[AI Study Guide]
  F --> G[academic_analyzer]
  F --> H[groq_client]
  F --> I[guidance_service]
  F --> J[tutor_service]
```

Modules:
- app.py: navigation and routing
- ai_study_guide/academic_analyzer.py: de-identified analysis
- ai_study_guide/groq_client.py: Groq API wrapper
- ai_study_guide/guidance_service.py: structured guidance
- ai_study_guide/study_guide_view.py: UI
- ai_study_guide/tutor_service.py: tutor chat
