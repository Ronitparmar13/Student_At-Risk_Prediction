import streamlit as st
import pandas as pd

from ai_study_guide.academic_analyzer import analyze_student_academics
from ai_study_guide.groq_client import GroqClient, GroqAuthenticationError, GroqRateLimitError
from ai_study_guide.guidance_service import generate_academic_guidance, GuidanceValidationError
from ai_study_guide.tutor_service import TutorService, MAX_MESSAGE_LENGTH

SUBJECT_LABELS = {
    'Math_Score': 'Mathematics',
    'Science_Score': 'Science',
    'English_Score': 'English',
    'Social_Studies_Score': 'Social Studies',
    'Business_Studies_Score': 'Business Studies',
    'Computer_Science_Score': 'Computer Science',
}

def friendly(subject):
    return SUBJECT_LABELS.get(subject, subject)


def _get_dataset(results, raw_df):
    if raw_df is not None:
        return raw_df
    if results is not None:
        return results
    return None


def render_study_guide(results=None, dataset=None, raw_df=None):
    st.title("AI Study Guide")
    df = _get_dataset(results, dataset if dataset is not None else raw_df)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info("Upload a dataset first.")
        return
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # student selection
    id_cols = [c for c in df.columns if "id" in c.lower() or "name" in c.lower()]
    label_col = id_cols[0] if id_cols else df.columns[0]
    labels = df[label_col].astype(str).tolist()
    selected_idx = st.selectbox("Select student", range(len(labels)), format_func=lambda i: labels[i])
    student_row = df.iloc[selected_idx].to_dict()
    peers = df.drop(index=df.index[selected_idx])
    # de-identified numeric only
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    student_record = {c: student_row.get(c) for c in numeric_cols}
    class_df = peers[numeric_cols] if numeric_cols else pd.DataFrame()
    analysis = analyze_student_academics(student_record, class_df)
    # display academic performance
    st.subheader("Academic Performance")
    sa = analysis.get("subject_analysis", {})
    for subj, info in sa.items():
        if not isinstance(info, dict):
            continue
        if info.get("available"):
            st.write(f"{friendly(subj)}: student {info.get('student_score')} vs class avg {info.get('class_average')} (diff {info.get('difference')})")
        else:
            st.write(f"{subj}: unavailable")
    # generation button
    cache_key = f"study_guide_{selected_idx}_{len(df)}"
    if st.button("Generate AI Study Guide"):
        try:
            client = GroqClient()
            result = generate_academic_guidance(analysis, client=client)
            st.session_state[cache_key] = result
            st.session_state["study_guide_selected"] = selected_idx
            st.session_state["study_guide_dataset_len"] = len(df)
        except Exception as e:
            st.error(str(e))
            return
    # reuse
    if cache_key in st.session_state:
        if st.session_state.get("study_guide_selected") != selected_idx or st.session_state.get("study_guide_dataset_len") != len(df):
            del st.session_state[cache_key]
        else:
            res = st.session_state[cache_key]
            st.subheader("Academic Summary")
            st.write(res.get("academic_summary"))
            st.subheader("Strengths")
            strengths = res.get("strengths", [])
            if not strengths:
                st.info("No subjects are currently above the peer-class average in the available data. A consistent improvement plan can help identify and build on emerging strengths.")
            else:
                for s in strengths:
                    if isinstance(s, dict):
                        st.markdown(f"**{friendly(s.get('subject'))}** — {s.get('explanation')}")
            st.subheader("Areas for Improvement")
            for a in res.get("areas_for_improvement", []):
                if isinstance(a, dict):
                    st.markdown(f"**{friendly(a.get('subject'))}**")
                    st.write(f"Student score: {a.get('student_score')} | Class average: {a.get('class_average')} | Difference: {a.get('score_gap')}")
                    st.write(f"Interpretation: {a.get('interpretation')}")
                    for r in a.get("practical_learning_recommendations", []):
                        st.write(f"- {r}")
            st.subheader("Seven-Day Study Plan")
            for day in res.get("study_plan", []):
                if isinstance(day, dict):
                    with st.expander(f"Day {day.get('day_number')} — {day.get('main_subject_or_learning_focus')}"):
                        st.write(f"Goal: {day.get('study_goal')}")
                        st.write("Activities:")
                        for act in day.get("suggested_activities", []):
                            st.write(f"- {act}")
                        st.write(f"Duration: {day.get('estimated_study_duration')}")
                        st.write(f"Expected outcome: {day.get('realistic_completion_outcome')}")
            st.subheader("Teacher Guidance")
            for t in res.get("teacher_guidance", []):
                st.write(t)
            st.subheader("Student Encouragement")
            st.write(res.get("student_encouragement"))
            st.subheader("Limitations")
            for l in res.get("limitations", []):
                st.write(l)
    _render_tutor(results, df, selected_idx, analysis)




def _render_tutor(results, df, selected_idx, analysis):
    st.subheader("AI Tutor")
    if "tutor_history" not in st.session_state:
        st.session_state["tutor_history"] = []
    if st.session_state.get("tutor_student_key") != (selected_idx, len(df)):
        st.session_state["tutor_history"] = []
        st.session_state["tutor_student_key"] = (selected_idx, len(df))
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    scores = {}
    sa = analysis.get("subject_analysis", {})
    for k, v in sa.items():
        if isinstance(v, dict):
            scores[k] = v
    subjects = list(sa.keys())
    subject_options = {friendly(x): x for x in (subjects if subjects else ["General"])}
    subject_label = st.selectbox("Subject", list(subject_options.keys()), key="tutor_subject")
    subject = subject_options[subject_label]
    for msg in st.session_state["tutor_history"]:
        if msg["role"] == "assistant":
            st.markdown(f"**Assistant:**")
            st.markdown(msg["content"])
        else:
            st.markdown(f"**You:** {msg['content']}")
    q = st.text_input("Ask a question", key="tutor_input", max_chars=MAX_MESSAGE_LENGTH)
    if st.button("Send", key="tutor_send"):
        if q.strip():
            st.session_state["tutor_history"].append({"role": "user", "content": q})
            try:
                svc = TutorService()
                ctx = {"subjects": subjects, "scores": scores}
                reply = svc.chat(q, history=st.session_state["tutor_history"][:-1], subject=subject, academic_context=ctx)
                st.session_state["tutor_history"] = st.session_state["tutor_history"][-20:]
                st.session_state["tutor_history"].append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(str(e))
    if st.button("Clear conversation", key="tutor_clear"):
        st.session_state["tutor_history"] = []

# aliases
render = render_study_guide
show_study_guide = render_study_guide
main = render_study_guide
