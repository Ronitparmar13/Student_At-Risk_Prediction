"""Project Notes page — methodology, model facts, limitations, and responsible use.

Entry point: ``render_project_notes(mode)``

This page documents:
  - Project objective
  - Dataset provenance (unverified; leakage advisory)
  - Verified model specification from the fitted pipeline
  - Data-cleaning methodology
  - Feature exclusion rationale
  - Probability calibration limitation
  - Responsible-use guidance

No historical or unverifiable evaluation metrics are published here.
The dataset provenance section clearly states that the exact
Risk_Score >= 50 correspondence *suggests* rule-generated labels;
this is not described as confirmed provenance.
"""
from __future__ import annotations

import streamlit as st

from config.theme import TOKENS, LIGHT
from services.model_service import ModelLoadError, get_streamlit_cached_pipeline


def render_project_notes(mode: str = "light") -> None:
    """Render the Project Notes page.

    Parameters
    ----------
    mode: Active theme.
    """
    t = TOKENS.get(mode, LIGHT)

    st.markdown(
        f'<h2 style="color:{t["text_primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:800;margin-bottom:4px;">Project Notes</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{t["text_secondary"]};margin-bottom:20px;">'
        "Methodology, verified model details, known limitations, and responsible-use guidance.</p>",
        unsafe_allow_html=True,
    )

    # ── 1. Project Objective ───────────────────────────────────────────────
    _section(t, "1 — Project Objective")
    st.markdown(
        """
This dashboard is an **educational prototype** built by Team 4 to explore
a student academic-risk dataset and demonstrate a classification pipeline
on a Streamlit interface.

The tool allows educators to:
- Upload and clean a student CSV.
- Explore data quality, distributions, and group-level patterns.
- Run single and batch predictions using a pre-trained Random Forest model.
- Review model-flagged records through a filtered queue.

This is not a clinical decision system. No prediction should be used as
the sole basis for a consequential decision about a student.
        """,
        unsafe_allow_html=False,
    )

    # ── 2. Dataset Provenance ──────────────────────────────────────────────
    _section(t, "2 — Dataset Provenance")
    st.warning(
        "**Unverified provenance.** "
        "The original dataset was supplied externally. "
        "An exact correspondence between `Risk_Score >= 50` and the `At_Risk` target "
        "column was found for every row in a previously inspected copy. "
        "This *suggests* the target label may have been rule-generated from "
        "`Risk_Score` rather than derived from genuine future student outcomes. "
        "This has **not been independently confirmed**. "
        "`Risk_Score` is excluded from model inputs as a precaution."
    )
    st.markdown(
        """
- **Reported dataset size:** approximately 50,000 rows × 30 columns (not independently verified).
- **Target column:** `At_Risk` — binary integer, 0 = not at risk, 1 = at risk.
- **Label-generation method:** unverified. The rule `Risk_Score >= 50 → At_Risk = 1` exactly reproduces the target in the inspected copy; the underlying cause has not been established.
- **Independent validation set:** does not exist. All evaluation was performed on a held-out split of the same dataset.
- **Implication:** the model may be learning a simple threshold rule rather than predicting genuine future risk. Treat all outputs as exploratory.
        """,
    )

    # ── 3. Verified Model Specification ───────────────────────────────────
    _section(t, "3 — Verified Model Specification")

    try:
        _, schema = get_streamlit_cached_pipeline()
        st.success("Model loaded successfully — details below are from the live pipeline.")
        st.markdown(
            f"""
| Property | Value |
|---|---|
| Model type | RandomForestClassifier |
| scikit-learn version (at save) | 1.6.1 |
| Preprocessing | `SimpleImputer(median)` → `StandardScaler()` (numeric); `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` (categorical) |
| Total input features | {len(schema.feature_names)} |
| Categorical features | {len(schema.categorical_features)}: {', '.join(schema.categorical_features)} |
| Numeric features | {len(schema.numeric_features)} |
| Class labels | 0 = not at risk, 1 = at risk |
| At-risk probability index | {schema.at_risk_class_index} (located dynamically via `classes_`) |
| Training split | 80 / 20 stratified, `random_state=42` |
| `predict_proba` supported | Yes |
            """,
        )

        with st.expander("Full feature list (26 features in fitted order)"):
            for i, feat in enumerate(schema.feature_names):
                feat_type = (
                    "Categorical — known values: "
                    + str(schema.category_values[feat])
                    if feat in schema.categorical_features
                    else "Numeric"
                )
                st.markdown(f"**{i:02d}. {feat}** — {feat_type}")

    except ModelLoadError as exc:
        st.error(
            f"Model could not be loaded for live introspection: {exc}\n\n"
            "Static specification is shown below."
        )
        _static_model_spec()

    # ── 4. Data Cleaning ──────────────────────────────────────────────────
    _section(t, "4 — Data-Cleaning Methodology")
    st.markdown(
        """
Cleaning follows the steps in `Team4_Student_At_Risk.ipynb` cells 5–8:

1. **Column and string normalisation** — leading/trailing whitespace stripped from column names and all string values.
2. **Title-case normalisation** — `Gender`, `Stream`, and `Higher_Education_Interest` are title-cased for consistency.
3. **Numeric coercion** — non-categorical columns are coerced to numeric; non-parseable values become NaN.
4. **Duplicate removal** — exact duplicate rows are removed.
5. **Range validation** — values outside expected ranges are set to NaN rather than deleting whole rows:
   - Percentages and `_Score` columns: 0–100
   - `Previous_Semester_GPA`: 0–10
   - Soft-skill columns (`Analytical_Thinking`, `Creativity_Level`, `Attention_to_Detail`, `Communication_Skills`): 1–10
   - `Age`: 14–100
6. **Target validation** — rows with missing or non-binary `At_Risk` are dropped for supervised analysis.
        """,
    )

    # ── 5. Feature Exclusions ─────────────────────────────────────────────
    _section(t, "5 — Feature Exclusions")
    st.markdown(
        """
The following columns are **never** passed to the model as inputs:

| Column | Reason |
|---|---|
| `At_Risk` | Target variable |
| `Risk_Score` | Target leakage — exactly reproduces `At_Risk` via `>= 50` in the inspected dataset |
| `Student_ID` | Row identifier — carries no predictive signal |
| `Gender` | Excluded in the original notebook; excluded here to avoid encoding a fairness-sensitive attribute without proper auditing |
        """,
    )

    # ── 6. Probability Calibration Limitation ─────────────────────────────
    _section(t, "6 — Probability Calibration Limitation")
    st.markdown(
        """
RandomForest `predict_proba` outputs are **uncalibrated**.

The probability displayed on prediction pages is the fraction of the forest's
decision trees that voted for class 1 (at-risk). This is **not**:

- A calibrated posterior probability of the student being at risk.
- Overall model accuracy.
- A guaranteed confidence score.

Calibrated probabilities require fitting a calibration layer (e.g., Platt scaling
or isotonic regression) on a held-out calibration set. No such calibration has
been performed for this model. All probabilities are therefore labelled
**"estimated model probability (uncalibrated)"**.
        """,
    )

    # ── 7. Responsible Use ────────────────────────────────────────────────
    _section(t, "7 — Responsible Use Guidance")
    st.info(
        """
**This application is an educational prototype.**

- Predictions are decision-support signals, not authoritative risk assessments.
- A qualified educator must review all flagged cases in their full academic and personal context.
- Do not use predictions as the sole basis for consequential decisions about students.
- The dataset provenance is unverified; the model may be reproducing a simple rule rather than predicting future outcomes.
- Group-level statistics are descriptive. They do not imply causal explanations or fairness assessments.
- Before any real-world deployment, this system would require independent dataset validation, model auditing, fairness analysis, and regulatory review.
        """,
    )

    # ── 8. Technical notes ────────────────────────────────────────────────
    _section(t, "8 — Technical Notes")
    st.markdown(
        """
- **Python:** 3.12 · **scikit-learn:** 1.6.1 (pinned for pkl compatibility) · **Streamlit:** 1.66
- **Model artifact:** `best_student_risk_model.pkl` (SHA-256: `bc1c582b...`)
- **Source notebook:** `Team4_Student_At_Risk.ipynb`
- **Architecture:** modular — `services/`, `components/`, `views/`, `config/`
- **Test suite:** 201+ tests (services, components, page logic)
        """,
    )


def _section(t: dict, title: str) -> None:
    """Render a themed section heading."""
    st.markdown(
        f'<h3 style="color:{t["primary"]};font-family:Manrope,sans-serif;'
        f'font-weight:700;margin-top:28px;margin-bottom:6px;">{title}</h3>',
        unsafe_allow_html=True,
    )


def _static_model_spec() -> None:
    st.markdown(
        """
| Property | Value (from notebook source) |
|---|---|
| Model type | RandomForestClassifier |
| scikit-learn (save version) | 1.6.1 |
| Input features | 26 (verified from `feature_names_in_`) |
| Categorical | `Stream` (Commerce / Humanities / Science), `Higher_Education_Interest` (High / Low / Medium) |
| At-risk class index | 1 (confirmed from `classes_` at verification time) |
| `predict_proba` | Yes — uncalibrated |
        """,
    )
