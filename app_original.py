from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Student Success | At-Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
.hero { background: linear-gradient(120deg,#111d3c 0%,#233c70 60%,#3156a4 100%);
        padding: 28px 32px; border-radius: 22px; color: white; margin-bottom: 22px; }
.hero h1 { font-family: 'Manrope',sans-serif; color:white; font-size:2.05rem; margin:0 0 8px 0; }
.hero p { color:#dbe5ff; margin:0; font-size:1.02rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; color:#91b4ff; font-weight:700; font-size:.72rem; margin-bottom:10px; }
.section-title { font-family:'Manrope',sans-serif; color:#172442; font-size:1.25rem; font-weight:800; margin: 8px 0 4px 0; }
.section-sub { color:#68748b; margin-bottom:14px; }
.metric-card { background:white; border:1px solid #e5eaf3; border-radius:16px; padding:18px 20px; }
.metric-label { color:#6c7890; font-size:.82rem; font-weight:600; }
.metric-value { color:#172442; font-family:'Manrope',sans-serif; font-size:1.75rem; font-weight:800; margin-top:5px; }
.metric-note { color:#78849a; font-size:.78rem; margin-top:4px; }
div[data-testid="stSidebar"] { background:#111d3c; }
div[data-testid="stSidebar"] * { color:#f3f6ff; }
div[data-testid="stSidebar"] .stCaption { color:#b7c5e5; }
.stButton button[kind="primary"] { background:#3156a4; border:0; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

TARGET = "At_Risk"
ID_COLS = {"Student_ID"}
LEAKAGE_COLS = {"Risk_Score"}

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def clean_dataset(raw: pd.DataFrame):
    """Conservative cleaning based on the supplied notebook's cleaning steps."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    category_cols = [c for c in ["Gender", "Stream", "Higher_Education_Interest"] if c in df.columns]
    for col in category_cols:
        df[col] = df[col].str.title()

    non_numeric = set(category_cols + (["Student_ID"] if "Student_ID" in df.columns else []))
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    range_rules = {}
    for col in df.columns:
        if "Percentage" in col or col.endswith("_Score") or col == "Risk_Score":
            range_rules[col] = (0, 100)
    if "Previous_Semester_GPA" in df.columns:
        range_rules["Previous_Semester_GPA"] = (0, 10)
    for col in ["Analytical_Thinking", "Creativity_Level", "Attention_to_Detail", "Communication_Skills"]:
        if col in df.columns:
            range_rules[col] = (1, 10)
    if "Age" in df.columns:
        range_rules["Age"] = (14, 100)

    invalid_counts = {}
    for col, (low, high) in range_rules.items():
        invalid = df[col].notna() & ~df[col].between(low, high)
        invalid_counts[col] = int(invalid.sum())
        df.loc[invalid, col] = np.nan

    if TARGET in df.columns:
        invalid_target = df[TARGET].notna() & ~df[TARGET].isin([0, 1])
        invalid_counts["At_Risk_non_binary"] = int(invalid_target.sum())
        df.loc[invalid_target, TARGET] = np.nan
        missing_target = int(df[TARGET].isna().sum())
        df = df.dropna(subset=[TARGET]).copy()
        df[TARGET] = df[TARGET].astype(int)
    else:
        missing_target = 0
    return df, duplicate_count, invalid_counts, missing_target

def target_leakage_check(df):
    if {"Risk_Score", TARGET}.issubset(df.columns) and len(df):
        derived = (df["Risk_Score"] >= 50).astype(int)
        return float((derived == df[TARGET]).mean())
    return None

def build_pipeline(X, model_name):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    else:
        model = RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
        )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def train_models(df, selected):
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")
    model_df = df.dropna(subset=[TARGET]).copy()
    y = model_df[TARGET].astype(int)
    excluded = {TARGET, "Student_ID", "Risk_Score", "Gender"}
    feature_cols = [c for c in model_df.columns if c not in excluded]
    X = model_df[feature_cols].copy()
    if X.shape[1] == 0:
        raise ValueError("No usable features remain after excluding target, ID, Risk_Score, and Gender.")
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("At_Risk must contain both classes, encoded as 0 and 1.")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two records for a stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    trained, rows = {}, []
    for name in selected:
        pipe = build_pipeline(X_train, name)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })
        trained[name] = pipe
    return trained, pd.DataFrame(rows), X_train, X_test, y_train, y_test

def show_metric_row(df):
    cols = st.columns(4)
    values = [
        ("Records", f"{len(df):,}", "After cleaning"),
        ("Columns", f"{df.shape[1]:,}", "Available fields"),
        ("Missing cells", f"{int(df.isna().sum().sum()):,}", "Still require attention"),
        ("Duplicate rows", f"{int(df.duplicated().sum()):,}", "Remaining after cleaning"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col:
            metric_card(label, value, note)

# Sidebar
with st.sidebar:
    st.markdown("## 🎓 Student Success")
    st.caption("At-Risk Analytics Workspace")
    page = st.radio("NAVIGATION", ["Overview", "Data Explorer", "Model Lab", "Predict a Student", "Project Notes"])
    st.divider()
    st.caption("Upload a CSV to unlock analysis and model tools.")
    uploaded = st.file_uploader("Student dataset (.csv)", type=["csv"], key="dataset_upload")

st.markdown("""
<div class="hero">
  <div class="eyebrow">TEAM 4 · STUDENT SUCCESS ANALYTICS</div>
  <h1>Student At-Risk Prediction</h1>
  <p>Explore student performance data, audit data quality, compare models, and review risk predictions.</p>
</div>
""", unsafe_allow_html=True)

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        cleaned_df, duplicate_count, invalid_counts, missing_target_count = clean_dataset(raw_df)
        st.session_state["raw_df"] = raw_df
        st.session_state["clean_df"] = cleaned_df
        st.session_state["clean_meta"] = {
            "duplicate_count": duplicate_count,
            "invalid_counts": invalid_counts,
            "missing_target_count": missing_target_count
        }
        # Reset model cache when data changes
        if st.session_state.get("uploaded_name") != uploaded.name:
            st.session_state.pop("trained", None)
            st.session_state.pop("metrics", None)
        st.session_state["uploaded_name"] = uploaded.name
    except Exception as exc:
        st.error(f"Could not read or clean this CSV: {exc}")
        st.stop()

df = st.session_state.get("clean_df")
meta = st.session_state.get("clean_meta", {})

if page == "Overview":
    st.markdown('<div class="section-title">Workspace overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A quick snapshot of the uploaded dataset and project workflow.</div>', unsafe_allow_html=True)
    if df is None:
        st.info("Upload your student dataset CSV from the sidebar to get started.")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("01 · Import", "Upload CSV", "Load your student records")
        with c2:
            metric_card("02 · Explore", "Audit data", "Missing values, distributions, and patterns")
        with c3:
            metric_card("03 · Evaluate", "Train models", "Compare recall, F1, and error counts")
        st.markdown("#### Expected data")
        st.write("The notebook expects a target column named `At_Risk` with values `0` (not at risk) and `1` (at risk). A `Student_ID` identifier and `Risk_Score` are excluded from model inputs.")
    else:
        show_metric_row(df)
        st.write("")
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("#### Dataset preview")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### At-risk distribution")
            if TARGET in df.columns:
                counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0).rename(index={0:"Not at risk",1:"At risk"})
                fig = px.pie(values=counts.values, names=counts.index, hole=.65,
                             color_discrete_sequence=["#3156a4", "#f28e5b"])
                fig.update_layout(margin=dict(t=15,b=10,l=10,r=10), legend_title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("The `At_Risk` target column is missing.")
        leakage = target_leakage_check(df)
        if leakage is not None and leakage == 1.0:
            st.warning("Leakage warning: `Risk_Score >= 50` matches `At_Risk` for every cleaned row. Risk_Score is excluded from model inputs; verify how the labels were generated.")
        elif leakage is not None:
            st.info(f"Risk_Score threshold match with At_Risk: {leakage:.1%}. This is a diagnostic, not proof of how the target was created.")
        st.markdown("#### Cleaning summary")
        st.write(f"Exact duplicate rows removed: **{meta.get('duplicate_count', 0):,}**")
        st.write(f"Rows removed for missing/invalid target: **{meta.get('missing_target_count', 0):,}**")
        st.write("Out-of-range numeric values are set to missing rather than deleting whole records.")

elif page == "Data Explorer":
    st.markdown('<div class="section-title">Data explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Inspect the cleaned data, data quality, and descriptive patterns.</div>', unsafe_allow_html=True)
    if df is None:
        st.info("Upload a CSV first.")
    else:
        show_metric_row(df)
        tab1, tab2, tab3, tab4 = st.tabs(["Preview & schema", "Data quality", "Distributions", "Group patterns"])
        with tab1:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download cleaned CSV", df.to_csv(index=False).encode("utf-8"), "student_at_risk_cleaned.csv", "text/csv")
            schema = pd.DataFrame({"Column": df.columns, "Data type": [str(t) for t in df.dtypes],
                                   "Missing": [int(df[c].isna().sum()) for c in df.columns],
                                   "Unique values": [int(df[c].nunique(dropna=False)) for c in df.columns]})
            st.dataframe(schema, use_container_width=True, hide_index=True)
        with tab2:
            quality = pd.DataFrame({
                "Data type": df.dtypes.astype(str),
                "Missing count": df.isna().sum(),
                "Missing %": (df.isna().mean()*100).round(2),
                "Unique values": df.nunique(dropna=False),
            }).sort_values("Missing count", ascending=False)
            st.dataframe(quality, use_container_width=True)
            if meta.get("invalid_counts"):
                st.markdown("##### Out-of-range values changed to missing")
                st.dataframe(pd.DataFrame([{"Column": k, "Invalid values": v} for k,v in meta["invalid_counts"].items()]), use_container_width=True, hide_index=True)
            st.download_button("Download quality report", quality.to_csv().encode("utf-8"), "student_at_risk_data_quality_report.csv", "text/csv")
        with tab3:
            numeric = df.select_dtypes(include=np.number).columns.tolist()
            numeric = [c for c in numeric if c not in [TARGET, "Risk_Score"]]
            if numeric:
                selected = st.selectbox("Choose a numeric field", numeric)
                fig = px.histogram(df, x=selected, nbins=30, marginal="box", color_discrete_sequence=["#3156a4"])
                fig.update_layout(margin=dict(t=20,b=20), bargap=.08)
                st.plotly_chart(fig, use_container_width=True)
                if TARGET in df.columns:
                    fig2 = px.box(df, x=TARGET, y=selected, color=TARGET,
                                  labels={TARGET:"At_Risk (0 = no, 1 = yes)"},
                                  color_discrete_sequence=["#3156a4","#f28e5b"])
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No numeric fields available for charting.")
        with tab4:
            categorical = [c for c in ["Stream", "Gender", "Higher_Education_Interest"] if c in df.columns]
            if TARGET in df.columns and categorical:
                selected_cat = st.selectbox("Group by", categorical)
                group = df.groupby(selected_cat, dropna=False)[TARGET].agg(At_Risk_Rate="mean", Records="size").reset_index()
                group["At-Risk Rate (%)"] = group["At_Risk_Rate"] * 100
                st.plotly_chart(px.bar(group, x=selected_cat, y="At-Risk Rate (%)", color_discrete_sequence=["#3156a4"]), use_container_width=True)
                st.dataframe(group.drop(columns="At_Risk_Rate").round(2), use_container_width=True, hide_index=True)
                st.caption("Group comparisons are descriptive and should not be interpreted as causal explanations or as proof of fairness.")
            else:
                st.info("Group patterns require At_Risk and at least one of Stream, Gender, or Higher_Education_Interest.")

elif page == "Model Lab":
    st.markdown('<div class="section-title">Model lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Train and compare models with preprocessing fitted only on the training split.</div>', unsafe_allow_html=True)
    if df is None:
        st.info("Upload a CSV first.")
    elif TARGET not in df.columns:
        st.error("Your CSV must include an `At_Risk` column.")
    else:
        leakage = target_leakage_check(df)
        if leakage == 1.0:
            st.error("The target exactly matches Risk_Score >= 50. Risk_Score is excluded, but perfect scores may still indicate label construction or other leakage. Treat results as exploratory.")
        st.write("Features excluded from training: `At_Risk` (target), `Student_ID` (identifier), `Risk_Score` (potential target leakage), and `Gender` (excluded in the original notebook).")
        selected = st.multiselect("Models to train", ["Logistic Regression", "Random Forest"], default=["Logistic Regression", "Random Forest"])
        if st.button("Train and evaluate", type="primary", disabled=not selected):
            with st.spinner("Training models and evaluating the held-out test set..."):
                try:
                    trained, metrics, X_train, X_test, y_train, y_test = train_models(df, selected)
                    st.session_state["trained"] = trained
                    st.session_state["metrics"] = metrics
                    st.session_state["evaluation_data"] = (X_test, y_test)
                    st.success("Training completed.")
                except Exception as exc:
                    st.error(str(exc))
        metrics = st.session_state.get("metrics")
        if metrics is not None:
            st.markdown("#### Held-out test performance")
            st.dataframe(metrics.sort_values("F1-Score", ascending=False).round(4), use_container_width=True, hide_index=True)
            best = metrics.loc[metrics["F1-Score"].idxmax()]
            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card("Best F1 model", best["Model"], "Highest test F1-score")
            with c2: metric_card("Recall", f'{best["Recall"]:.1%}', "At-risk cases found")
            with c3: metric_card("Precision", f'{best["Precision"]:.1%}', "Predicted at-risk cases correct")
            with c4: metric_card("F1-score", f'{best["F1-Score"]:.3f}', "Precision/recall balance")
            fig = px.bar(metrics.melt(id_vars="Model", value_vars=["Precision","Recall","F1-Score"], var_name="Metric", value_name="Score"),
                         x="Metric", y="Score", color="Model", barmode="group", range_y=[0,1.05])
            st.plotly_chart(fig, use_container_width=True)
            X_test, y_test = st.session_state["evaluation_data"]
            trained = st.session_state["trained"]
            model_name = st.selectbox("Confusion matrix model", list(trained.keys()))
            pred = trained[model_name].predict(X_test)
            cm = confusion_matrix(y_test, pred, labels=[0,1])
            fig_cm = px.imshow(cm, text_auto=True, x=["Predicted: Not at risk","Predicted: At risk"],
                               y=["Actual: Not at risk","Actual: At risk"], color_continuous_scale="Blues",
                               labels=dict(x="Prediction", y="Actual", color="Count"))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.code(classification_report(y_test, pred, target_names=["Not at risk","At risk"], zero_division=0))
            model_bytes = None
            if model_name in trained:
                import tempfile
                temp_path = Path(tempfile.gettempdir()) / "best_student_risk_model.pkl"
                joblib.dump(trained[model_name], temp_path)
                model_bytes = temp_path.read_bytes()
            if model_bytes:
                st.download_button("Download selected trained model", model_bytes, f"{model_name.lower().replace(' ','_')}_student_risk.joblib", "application/octet-stream")
            st.warning("Model scores are not proof of real-world performance. Validate against an independent dataset before using predictions to make student-support decisions.")

elif page == "Predict a Student":
    st.markdown('<div class="section-title">Predict a student</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Use a trained model to estimate whether a record resembles the at-risk class.</div>', unsafe_allow_html=True)
    trained = st.session_state.get("trained")
    if not trained or df is None:
        st.info("Upload a dataset, then train at least one model in Model Lab.")
    else:
        model_name = st.selectbox("Model", list(trained.keys()))
        model = trained[model_name]
        # Feature list mirrors the exact training exclusions.
        feature_cols = [c for c in df.columns if c not in {TARGET, "Student_ID", "Risk_Score", "Gender"}]
        values = {}
        with st.form("prediction_form"):
            cols = st.columns(2)
            for i, col in enumerate(feature_cols):
                target_col = cols[i % 2]
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    median = series.median()
                    if pd.isna(median): median = 0.0
                    minv = float(series.min()) if series.notna().any() else 0.0
                    maxv = float(series.max()) if series.notna().any() else max(minv + 1, 1)
                    if minv == maxv: maxv = minv + 1
                    values[col] = target_col.number_input(col.replace("_"," "), value=float(median), min_value=minv, max_value=maxv)
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist())
                    values[col] = target_col.selectbox(col.replace("_"," "), options if options else ["Unknown"])
            submitted = st.form_submit_button("Estimate risk", type="primary")
        if submitted:
            input_df = pd.DataFrame([values], columns=feature_cols)
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0,1])
            left,right = st.columns([1,2])
            with left:
                metric_card("Model estimate", "AT RISK" if prediction == 1 else "NOT AT RISK",
                            "For review only — not a diagnosis or final decision")
            with right:
                st.markdown("#### Estimated at-risk probability")
                st.progress(max(0, min(100, int(probability*100))))
                st.write(f"**{probability:.1%}** estimated probability for class 1.")
            st.warning("Use this output only as a decision-support signal. Review context with educators and the student; do not use it as the sole basis for consequential decisions.")

elif page == "Project Notes":
    st.markdown('<div class="section-title">Project notes & limitations</div>', unsafe_allow_html=True)
    st.markdown("""
**Purpose:** explore student data quality and evaluate baseline classifiers for the `At_Risk` target.

**Cleaning performed**
- Trims column and text whitespace; normalizes selected category labels.
- Converts expected non-ID fields to numeric where possible.
- Removes exact duplicate rows.
- Flags known out-of-range values by changing them to missing.
- Drops records with missing/invalid target labels for supervised training.

**Modeling safeguards**
- Train/test split is stratified and occurs before imputers, encoders, and scalers are fitted.
- `At_Risk`, `Student_ID`, `Risk_Score`, and `Gender` are not model features.
- Unknown categorical values at prediction time are handled by one-hot encoding with `handle_unknown="ignore"`.

**Important limitations**
- The notebook describes a possible exact relationship between `Risk_Score >= 50` and `At_Risk`. If the label was derived from that rule, the task may be reproducing a generated label rather than predicting future risk.
- Perfect test performance can result from leakage, duplicates, label construction, or an unusually easy/synthetic dataset. It should be investigated rather than celebrated as proof of generalization.
- Correlations and group differences are descriptive, not causal.
- This is a prototype. A qualified human should review any student-support decision.
""")
    st.markdown("**Expected target:** `At_Risk` with binary values `0` and `1`.")
    st.markdown("**Excluded inputs:** `Student_ID`, `Risk_Score`, and `Gender`.")

st.markdown("---")
st.caption("Team 4 · Student Success & At-Risk Prediction · Prototype for educational use")
