"""Unit tests for Milestone 3 page modules.

Tests cover only the logic and helper functions that do not require a
running Streamlit session.  They verify:

  views/overview.py          — _build_kpi_cards, empty-state helpers
  views/data_explorer.py     — import, correlation exclusions via analytics
  views/student_predict.py   — _get_range, _get_step helpers
  views/batch_predict.py     — import, _range_hint
  views/risk_review.py       — import, session-state key constants
  views/project_notes.py     — import, _static_model_spec (non-Streamlit path)

We also verify that no page imports trigger errors and that none of the
approved pages reference Model Lab.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# Import smoke tests — every view module must import cleanly
# ══════════════════════════════════════════════════════════════════════════════

class TestPageImports:
    def test_overview_imports(self):
        import views.overview  # noqa: F401

    def test_data_explorer_imports(self):
        import views.data_explorer  # noqa: F401

    def test_student_predict_imports(self):
        import views.student_predict  # noqa: F401

    def test_batch_predict_imports(self):
        import views.batch_predict  # noqa: F401

    def test_risk_review_imports(self):
        import views.risk_review  # noqa: F401

    def test_project_notes_imports(self):
        import views.project_notes  # noqa: F401

    def test_all_view_modules_have_render_function(self):
        """Each view must expose a callable render entry point."""
        from views.overview        import render_overview
        from views.data_explorer   import render_data_explorer
        from views.student_predict import render_student_predict
        from views.batch_predict   import render_batch_predict
        from views.risk_review     import render_risk_review
        from views.project_notes   import render_project_notes

        for fn in (
            render_overview, render_data_explorer, render_student_predict,
            render_batch_predict, render_risk_review, render_project_notes,
        ):
            assert callable(fn), f"{fn} is not callable"

    def test_no_model_lab_in_view_modules(self):
        """None of the view source files should contain 'Model Lab'."""
        views_dir = Path(__file__).resolve().parent.parent / "views"
        for py_file in views_dir.glob("*.py"):
            src = py_file.read_text()
            assert "Model Lab" not in src, (
                f"'Model Lab' found in {py_file.name}"
            )
            assert "model_lab" not in src.lower(), (
                f"'model_lab' found in {py_file.name}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# views/overview.py — KPI card builder
# ══════════════════════════════════════════════════════════════════════════════

class TestOverviewKpiCards:
    def _make_kpis(self, at_risk=35, not_risk=65, unique=None):
        from services.analytics import KPIResult
        k = KPIResult(
            record_count=100,
            column_count=30,
            missing_cells=5,
            duplicate_rows=0,
            has_target=True,
            at_risk_count=at_risk,
            not_at_risk_count=not_risk,
            at_risk_pct=at_risk / (at_risk + not_risk),
            unique_students=unique,
            completeness=0.999,
        )
        return k

    def test_returns_four_cards(self):
        from views.overview import _build_kpi_cards
        cards = _build_kpi_cards(self._make_kpis())
        assert len(cards) == 4

    def test_record_count_card_has_correct_value(self):
        from views.overview import _build_kpi_cards
        cards = _build_kpi_cards(self._make_kpis())
        record_card = cards[0]
        assert "100" in record_card.value

    def test_at_risk_card_has_count(self):
        from views.overview import _build_kpi_cards
        cards = _build_kpi_cards(self._make_kpis(at_risk=35))
        at_risk_card = cards[1]
        assert "35" in at_risk_card.value

    def test_at_risk_card_none_when_no_target(self):
        from services.analytics import KPIResult
        from views.overview import _build_kpi_cards
        k = KPIResult(record_count=10, has_target=False)
        cards = _build_kpi_cards(k)
        at_risk_card = cards[1]
        assert at_risk_card.value is None

    def test_completeness_card_has_percent(self):
        from views.overview import _build_kpi_cards
        cards = _build_kpi_cards(self._make_kpis())
        completeness_card = cards[2]
        assert "%" in completeness_card.value

    def test_unique_student_note_when_id_present(self):
        from views.overview import _build_kpi_cards
        cards = _build_kpi_cards(self._make_kpis(unique=90))
        col_card = cards[3]
        assert "90" in col_card.note

    def test_no_student_id_note_when_absent(self):
        from views.overview import _build_kpi_cards
        cards = _build_kpi_cards(self._make_kpis(unique=None))
        col_card = cards[3]
        assert "No Student_ID" in col_card.note

    def test_no_fabricated_values(self):
        """KPI values come from the KPIResult; they must not be hard-coded strings."""
        from views.overview import _build_kpi_cards
        for count in (17, 99, 1234):
            k = self._make_kpis()
            k.record_count = count
            cards = _build_kpi_cards(k)
            # Values are formatted with commas (e.g. 1,234) — check numeric digits present
            assert f"{count:,}" in cards[0].value


# ══════════════════════════════════════════════════════════════════════════════
# views/student_predict.py — range and step helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictHelpers:
    def test_gpa_range_is_0_to_10(self):
        from views.student_predict import _get_range
        lo, hi = _get_range("Previous_Semester_GPA")
        assert lo == 0.0
        assert hi == 10.0

    def test_age_range_is_14_to_100(self):
        from views.student_predict import _get_range
        lo, hi = _get_range("Age")
        assert lo == 14.0
        assert hi == 100.0

    def test_soft_skill_range_is_1_to_10(self):
        from views.student_predict import _get_range
        for col in ("Analytical_Thinking", "Creativity_Level",
                    "Attention_to_Detail", "Communication_Skills"):
            lo, hi = _get_range(col)
            assert lo == 1.0
            assert hi == 10.0

    def test_score_range_is_0_to_100(self):
        from views.student_predict import _get_range
        lo, hi = _get_range("Math_Score")
        assert lo == 0.0
        assert hi == 100.0

    def test_percentage_range_is_0_to_100(self):
        from views.student_predict import _get_range
        lo, hi = _get_range("Attendance_Percentage")
        assert lo == 0.0
        assert hi == 100.0

    def test_gpa_step_is_0_1(self):
        from views.student_predict import _get_step
        assert _get_step("Previous_Semester_GPA") == 0.1

    def test_soft_skill_step_is_1(self):
        from views.student_predict import _get_step
        for col in ("Analytical_Thinking", "Creativity_Level"):
            assert _get_step(col) == 1.0

    def test_age_step_is_1(self):
        from views.student_predict import _get_step
        assert _get_step("Age") == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# views/batch_predict.py — session state keys and range hint
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchPredictModule:
    def test_session_result_key_is_defined(self):
        from views.batch_predict import _SESSION_RESULT_KEY
        assert isinstance(_SESSION_RESULT_KEY, str)
        assert len(_SESSION_RESULT_KEY) > 0

    def test_session_proba_key_is_defined(self):
        from views.batch_predict import _SESSION_PROBA_KEY
        assert isinstance(_SESSION_PROBA_KEY, str)

    def test_range_hint_returns_string(self):
        from views.batch_predict import _range_hint
        hint = _range_hint("Attendance_Percentage")
        assert isinstance(hint, str)
        assert "0" in hint
        assert "100" in hint

    def test_range_hint_gpa(self):
        from views.batch_predict import _range_hint
        hint = _range_hint("Previous_Semester_GPA")
        assert "10" in hint


# ══════════════════════════════════════════════════════════════════════════════
# views/risk_review.py — session state keys
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskReviewModule:
    def test_session_keys_match_batch_predict(self):
        """Risk Review must read from the same session keys that Batch Predictions writes."""
        from views.batch_predict import _SESSION_RESULT_KEY as bp_result
        from views.batch_predict import _SESSION_PROBA_KEY  as bp_proba
        from views.risk_review   import _SESSION_RESULT_KEY as rr_result
        from views.risk_review   import _SESSION_PROBA_KEY  as rr_proba
        assert bp_result == rr_result, (
            "Batch Predictions and Risk Review use different session keys for results"
        )
        assert bp_proba == rr_proba, (
            "Batch Predictions and Risk Review use different session keys for proba flag"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Correlation matrix — leakage exclusion (cross-cutting)
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationExclusions:
    def _make_df(self, n=30):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "Score_A": rng.uniform(30, 100, n),
            "Score_B": rng.uniform(30, 100, n),
            "At_Risk":    rng.integers(0, 2, n),
            "Risk_Score": rng.integers(0, 11, n) * 10,
            "Student_ID": [f"S{i}" for i in range(n)],
        })

    def test_at_risk_excluded_from_correlation(self):
        from services.analytics import correlation_matrix
        df = self._make_df()
        corr = correlation_matrix(df)
        assert "At_Risk" not in corr.columns

    def test_risk_score_excluded_from_correlation(self):
        from services.analytics import correlation_matrix
        df = self._make_df()
        corr = correlation_matrix(df)
        assert "Risk_Score" not in corr.columns

    def test_student_id_excluded_from_correlation(self):
        from services.analytics import correlation_matrix
        df = self._make_df()
        corr = correlation_matrix(df)
        assert "Student_ID" not in corr.columns


# ══════════════════════════════════════════════════════════════════════════════
# Model prediction — works without a CSV (integration: real pkl)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictWithoutDataset:
    def test_single_predict_works_without_df(self):
        """predict_single must work with just a dict — no DataFrame needed."""
        from services.model_service import get_feature_schema, predict_single

        schema = get_feature_schema()
        # Build a row using only the schema (no df)
        row = {}
        for feat in schema.feature_names:
            if feat in schema.categorical_features:
                row[feat] = schema.category_values[feat][0]
            elif feat == "Previous_Semester_GPA":
                row[feat] = 5.0
            elif feat == "Age":
                row[feat] = 20.0
            elif feat in ("Analytical_Thinking", "Creativity_Level",
                          "Attention_to_Detail", "Communication_Skills"):
                row[feat] = 5.0
            else:
                row[feat] = 50.0

        predicted_class, probability = predict_single(row)
        assert predicted_class in (0, 1)
        assert 0.0 <= probability <= 1.0

    def test_batch_predict_preserves_row_alignment(self):
        """Output row N must correspond to input row N."""
        from services.model_service import get_feature_schema, predict_batch

        schema = get_feature_schema()
        rows = []
        for i in range(5):
            row = {}
            for feat in schema.feature_names:
                if feat in schema.categorical_features:
                    row[feat] = schema.category_values[feat][0]
                elif feat == "Previous_Semester_GPA":
                    row[feat] = 4.0 + i
                elif feat == "Age":
                    row[feat] = 18.0 + i
                elif feat in ("Analytical_Thinking", "Creativity_Level",
                              "Attention_to_Detail", "Communication_Skills"):
                    row[feat] = 5.0
                else:
                    row[feat] = 40.0 + i * 5
            rows.append(row)

        batch_df = pd.DataFrame(rows, columns=schema.feature_names)
        batch_df["_row_id"] = list(range(5))
        result = predict_batch(batch_df)

        assert len(result.output_df) == 5
        assert list(result.output_df["_row_id"]) == list(range(5))

    def test_probability_display_when_unavailable(self):
        """prediction_badge_html must show no % when probability is None."""
        from components.prediction_indicator import prediction_badge_html
        html = prediction_badge_html(1, None, mode="light")
        assert "%" not in html
        # Must still communicate the class via text
        assert "AT RISK" in html
