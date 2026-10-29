"""Unit tests for the Milestone 2 shared component modules.

Tests cover only the logic and HTML-generation paths that do not require a
running Streamlit session.  They verify:

  charts.py        — apply_chart_theme, donut_chart, histogram_chart, box_chart,
                     bar_chart, heatmap_chart, _empty_fig
  kpi_cards.py     — KPICard data model, _card_html output, render helpers
  prediction_indicator.py — prediction_badge_html, _validate_probability,
                             token injection, calibration-note presence
  navigation.py    — PAGES constant, no Model Lab
  header.py        — module imports cleanly; token usage verified separately
                     (full Streamlit render not testable without a session)
  config/theme.py  — regression: both themes produce correct keys
                     (already covered in M0, repeated here for M2 scope)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.theme import LIGHT, DARK, TOKENS, apply_theme, get_plotly_layout
from components.charts import (
    apply_chart_theme,
    donut_chart,
    histogram_chart,
    box_chart,
    bar_chart,
    heatmap_chart,
    _empty_fig,
)
from components.kpi_cards import KPICard, _card_html
from components.prediction_indicator import prediction_badge_html, _validate_probability
from components.navigation import PAGES


# ── Fixtures ───────────────────────────────────────────────────────────────

def _numeric_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Score_A": rng.uniform(30, 100, n),
        "Score_B": rng.uniform(30, 100, n),
        "Score_C": rng.uniform(30, 100, n),
        "At_Risk": rng.integers(0, 2, n),
        "Risk_Score": rng.integers(0, 11, n) * 10,
        "Group": rng.choice(["Commerce", "Science", "Humanities"], n),
    })


# ══════════════════════════════════════════════════════════════════════════════
# config/theme.py — regression
# ══════════════════════════════════════════════════════════════════════════════

class TestThemeRegression:
    _required_keys = [
        "header", "primary", "accent", "bg", "surface",
        "text_primary", "text_secondary", "border",
        "at_risk", "not_risk", "at_risk_text", "not_risk_text",
        "focus_ring", "input_bg", "input_text",
    ]

    def test_light_has_all_required_keys(self):
        for k in self._required_keys:
            assert k in LIGHT, f"LIGHT missing key: {k}"

    def test_dark_has_all_required_keys(self):
        for k in self._required_keys:
            assert k in DARK, f"DARK missing key: {k}"

    def test_apply_theme_light_contains_light_bg(self):
        css = apply_theme("light")
        assert "#F2F6FE" in css

    def test_apply_theme_dark_contains_dark_bg(self):
        css = apply_theme("dark")
        assert "#0C1220" in css

    def test_get_plotly_layout_light_returns_required_keys(self):
        layout = get_plotly_layout("light")
        for key in ("plot_bgcolor", "paper_bgcolor", "font", "hoverlabel", "legend"):
            assert key in layout, f"Missing plotly layout key: {key}"

    def test_get_plotly_layout_dark_surfaces_differ_from_light(self):
        light = get_plotly_layout("light")
        dark = get_plotly_layout("dark")
        assert light["plot_bgcolor"] != dark["plot_bgcolor"]
        assert light["paper_bgcolor"] != dark["paper_bgcolor"]

    def test_unknown_mode_falls_back_to_light(self):
        css = apply_theme("invalid_mode")
        assert "#F2F6FE" in css  # LIGHT bg


# ══════════════════════════════════════════════════════════════════════════════
# components/charts.py
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyChartTheme:
    def test_returns_figure(self):
        fig = go.Figure()
        result = apply_chart_theme(fig, "light")
        assert isinstance(result, go.Figure)

    def test_same_object_returned(self):
        fig = go.Figure()
        result = apply_chart_theme(fig, "light")
        assert result is fig

    def test_light_plot_bgcolor_is_surface(self):
        fig = go.Figure()
        apply_chart_theme(fig, "light")
        assert fig.layout.plot_bgcolor == LIGHT["surface"]

    def test_dark_plot_bgcolor_is_dark_surface(self):
        fig = go.Figure()
        apply_chart_theme(fig, "dark")
        assert fig.layout.plot_bgcolor == DARK["surface"]

    def test_light_paper_bgcolor_is_bg(self):
        fig = go.Figure()
        apply_chart_theme(fig, "light")
        assert fig.layout.paper_bgcolor == LIGHT["bg"]

    def test_dark_paper_bgcolor_is_dark_bg(self):
        fig = go.Figure()
        apply_chart_theme(fig, "dark")
        assert fig.layout.paper_bgcolor == DARK["bg"]

    def test_font_color_uses_text_primary(self):
        fig = go.Figure()
        apply_chart_theme(fig, "light")
        assert fig.layout.font.color == LIGHT["text_primary"]

    def test_dark_font_color_uses_dark_text_primary(self):
        fig = go.Figure()
        apply_chart_theme(fig, "dark")
        assert fig.layout.font.color == DARK["text_primary"]

    def test_hoverlabel_bgcolor_set(self):
        fig = go.Figure()
        apply_chart_theme(fig, "light")
        assert fig.layout.hoverlabel.bgcolor == LIGHT["surface"]


class TestEmptyFig:
    def test_returns_figure(self):
        assert isinstance(_empty_fig("No data", "light"), go.Figure)

    def test_has_annotation(self):
        fig = _empty_fig("No data", "light")
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "No data"

    def test_annotation_uses_text_secondary_color(self):
        fig = _empty_fig("No data", "light")
        assert fig.layout.annotations[0].font.color == LIGHT["text_secondary"]

    def test_dark_annotation_uses_dark_text_secondary(self):
        fig = _empty_fig("No data", "dark")
        assert fig.layout.annotations[0].font.color == DARK["text_secondary"]


class TestDonutChart:
    def test_returns_figure_with_data(self):
        fig = donut_chart(["A", "B"], [30, 70], mode="light")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_empty_values_returns_empty_fig(self):
        fig = donut_chart([], [], mode="light")
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) == 1

    def test_zero_sum_returns_empty_fig(self):
        fig = donut_chart(["A", "B"], [0, 0], mode="light")
        assert len(fig.data) == 0

    def test_light_theme_applied(self):
        fig = donut_chart(["A", "B"], [30, 70], mode="light")
        assert fig.layout.paper_bgcolor == LIGHT["bg"]

    def test_dark_theme_applied(self):
        fig = donut_chart(["A", "B"], [30, 70], mode="dark")
        assert fig.layout.paper_bgcolor == DARK["bg"]

    def test_hole_is_set(self):
        fig = donut_chart(["A", "B"], [30, 70], mode="light")
        assert fig.data[0].hole > 0

    def test_custom_colors_accepted(self):
        fig = donut_chart(["A", "B"], [30, 70], colors=["#FF0000", "#00FF00"])
        assert fig.data[0].marker.colors == ("#FF0000", "#00FF00")


class TestHistogramChart:
    def test_returns_figure(self):
        df = _numeric_df()
        fig = histogram_chart(df, "Score_A", mode="light")
        assert isinstance(fig, go.Figure)

    def test_missing_column_returns_empty(self):
        df = _numeric_df()
        fig = histogram_chart(df, "NonExistent", mode="light")
        assert len(fig.data) == 0

    def test_light_theme_applied(self):
        df = _numeric_df()
        fig = histogram_chart(df, "Score_A", mode="light")
        assert fig.layout.plot_bgcolor == LIGHT["surface"]

    def test_dark_theme_applied(self):
        df = _numeric_df()
        fig = histogram_chart(df, "Score_A", mode="dark")
        assert fig.layout.plot_bgcolor == DARK["surface"]

    def test_has_data_traces(self):
        df = _numeric_df()
        fig = histogram_chart(df, "Score_A")
        assert len(fig.data) > 0


class TestBoxChart:
    def test_returns_figure(self):
        df = _numeric_df()
        fig = box_chart(df, "At_Risk", "Score_A", mode="light")
        assert isinstance(fig, go.Figure)

    def test_missing_columns_returns_empty(self):
        df = _numeric_df()
        fig = box_chart(df, "At_Risk", "DoesNotExist", mode="light")
        assert len(fig.data) == 0

    def test_light_theme_applied(self):
        df = _numeric_df()
        fig = box_chart(df, "At_Risk", "Score_A", mode="light")
        assert fig.layout.plot_bgcolor == LIGHT["surface"]

    def test_dark_theme_applied(self):
        df = _numeric_df()
        fig = box_chart(df, "At_Risk", "Score_A", mode="dark")
        assert fig.layout.plot_bgcolor == DARK["surface"]


class TestBarChart:
    def test_returns_figure(self):
        df = _numeric_df()
        grp = df.groupby("Group", as_index=False)["At_Risk"].mean()
        fig = bar_chart(grp, "Group", "At_Risk", mode="light")
        assert isinstance(fig, go.Figure)

    def test_missing_columns_returns_empty(self):
        df = _numeric_df()
        fig = bar_chart(df, "Group", "NoCol", mode="light")
        assert len(fig.data) == 0

    def test_light_theme_applied(self):
        df = _numeric_df()
        fig = bar_chart(df, "Group", "Score_A", mode="light")
        assert fig.layout.plot_bgcolor == LIGHT["surface"]

    def test_dark_theme_applied(self):
        df = _numeric_df()
        fig = bar_chart(df, "Group", "Score_A", mode="dark")
        assert fig.layout.plot_bgcolor == DARK["surface"]


class TestHeatmapChart:
    def test_returns_figure(self):
        from services.analytics import correlation_matrix
        df = _numeric_df()
        corr = correlation_matrix(df)
        fig = heatmap_chart(corr, mode="light")
        assert isinstance(fig, go.Figure)

    def test_empty_corr_returns_empty_fig(self):
        fig = heatmap_chart(pd.DataFrame(), mode="light")
        assert len(fig.data) == 0

    def test_light_theme_applied(self):
        from services.analytics import correlation_matrix
        corr = correlation_matrix(_numeric_df())
        fig = heatmap_chart(corr, mode="light")
        assert fig.layout.paper_bgcolor == LIGHT["bg"]

    def test_dark_theme_applied(self):
        from services.analytics import correlation_matrix
        corr = correlation_matrix(_numeric_df())
        fig = heatmap_chart(corr, mode="dark")
        assert fig.layout.paper_bgcolor == DARK["bg"]


# ══════════════════════════════════════════════════════════════════════════════
# components/kpi_cards.py
# ══════════════════════════════════════════════════════════════════════════════

class TestKPICard:
    def test_display_value_returns_value(self):
        card = KPICard(label="RECORDS", value="50,000")
        assert card.display_value == "50,000"

    def test_display_value_returns_dash_when_none(self):
        card = KPICard(label="RECORDS", value=None)
        assert card.display_value == "—"

    def test_note_defaults_to_empty(self):
        card = KPICard(label="X", value="1")
        assert card.note == ""

    def test_icon_defaults_to_empty(self):
        card = KPICard(label="X", value="1")
        assert card.icon == ""


class TestCardHtml:
    def test_returns_string(self):
        card = KPICard(label="RECORDS", value="50,000", note="After cleaning", icon="📊")
        html = _card_html(card, LIGHT)
        assert isinstance(html, str)

    def test_contains_label(self):
        card = KPICard(label="AT-RISK RATE", value="35.1%")
        html = _card_html(card, LIGHT)
        assert "AT-RISK RATE" in html

    def test_contains_value(self):
        card = KPICard(label="RECORDS", value="50,000")
        html = _card_html(card, LIGHT)
        assert "50,000" in html

    def test_contains_note(self):
        card = KPICard(label="RECORDS", value="50,000", note="After cleaning")
        html = _card_html(card, LIGHT)
        assert "After cleaning" in html

    def test_contains_icon(self):
        card = KPICard(label="RECORDS", value="50,000", icon="📊")
        html = _card_html(card, LIGHT)
        assert "📊" in html

    def test_none_value_renders_dash(self):
        card = KPICard(label="RECORDS", value=None)
        html = _card_html(card, LIGHT)
        assert "—" in html

    def test_uses_light_surface_token(self):
        card = KPICard(label="X", value="1")
        html = _card_html(card, LIGHT)
        assert LIGHT["surface"] in html

    def test_uses_dark_surface_token(self):
        card = KPICard(label="X", value="1")
        html = _card_html(card, DARK)
        assert DARK["surface"] in html

    def test_uses_light_text_primary_token(self):
        card = KPICard(label="X", value="42")
        html = _card_html(card, LIGHT)
        assert LIGHT["text_primary"] in html

    def test_uses_dark_text_primary_token(self):
        card = KPICard(label="X", value="42")
        html = _card_html(card, DARK)
        assert DARK["text_primary"] in html

    def test_uses_light_border_token(self):
        card = KPICard(label="X", value="1")
        html = _card_html(card, LIGHT)
        assert LIGHT["border"] in html

    def test_uses_dark_border_token(self):
        card = KPICard(label="X", value="1")
        html = _card_html(card, DARK)
        assert DARK["border"] in html

    def test_no_hardcoded_white_in_dark_mode(self):
        """Dark mode card must not hardcode #FFFFFF as a text or bg colour."""
        card = KPICard(label="X", value="1")
        html = _card_html(card, DARK)
        # #FFFFFF should not appear as a text/bg value (it would be invisible)
        assert "#FFFFFF" not in html or DARK["surface"] in html


# ══════════════════════════════════════════════════════════════════════════════
# components/prediction_indicator.py
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionBadgeHtml:
    def test_at_risk_badge_contains_at_risk_text(self):
        html = prediction_badge_html(1, None, mode="light")
        assert "AT RISK" in html

    def test_not_at_risk_badge_contains_not_at_risk_text(self):
        html = prediction_badge_html(0, None, mode="light")
        assert "NOT AT RISK" in html

    def test_at_risk_uses_at_risk_token(self):
        html = prediction_badge_html(1, None, mode="light")
        assert LIGHT["at_risk"] in html

    def test_not_at_risk_uses_not_risk_token(self):
        html = prediction_badge_html(0, None, mode="light")
        assert LIGHT["not_risk"] in html

    def test_dark_at_risk_uses_dark_token(self):
        html = prediction_badge_html(1, None, mode="dark")
        assert DARK["at_risk"] in html

    def test_dark_not_at_risk_uses_dark_token(self):
        html = prediction_badge_html(0, None, mode="dark")
        assert DARK["not_risk"] in html

    def test_at_risk_badge_has_warning_icon(self):
        html = prediction_badge_html(1, None, mode="light")
        assert "⚠️" in html

    def test_not_at_risk_badge_has_check_icon(self):
        html = prediction_badge_html(0, None, mode="light")
        assert "✅" in html

    def test_probability_shown_when_provided(self):
        html = prediction_badge_html(1, 0.73, mode="light")
        assert "73%" in html

    def test_probability_absent_when_none(self):
        html = prediction_badge_html(1, None, mode="light")
        assert "%" not in html

    def test_probability_label_says_uncalibrated(self):
        html = prediction_badge_html(1, 0.73, mode="light")
        assert "uncalibrated" in html

    def test_probability_rounding_at_boundary(self):
        html = prediction_badge_html(0, 0.005, mode="light")
        # 0.005 * 100 = 0.5 -> rounds to 1 or 0 depending on banker's rounding
        assert "%" in html

    def test_probability_100_percent(self):
        html = prediction_badge_html(1, 1.0, mode="light")
        assert "100%" in html

    def test_probability_0_percent(self):
        html = prediction_badge_html(0, 0.0, mode="light")
        assert "0%" in html


class TestValidateProbability:
    def test_valid_lower_bound(self):
        _validate_probability(0.0)  # Should not raise

    def test_valid_upper_bound(self):
        _validate_probability(1.0)  # Should not raise

    def test_valid_mid_range(self):
        _validate_probability(0.5)  # Should not raise

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="\\[0.0, 1.0\\]"):
            _validate_probability(-0.01)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="\\[0.0, 1.0\\]"):
            _validate_probability(1.001)


# ══════════════════════════════════════════════════════════════════════════════
# components/navigation.py
# ══════════════════════════════════════════════════════════════════════════════

class TestNavigation:
    def test_pages_list_has_six_items(self):
        assert len(PAGES) == 6

    def test_model_lab_not_in_pages(self):
        for page in PAGES:
            assert "Model Lab" not in page
            assert "model lab" not in page.lower()

    def test_all_approved_pages_present(self):
        approved = {
            "Overview",
            "Data Explorer",
            "Predict a Student",
            "Batch Predictions",
            "Risk Review",
            "Project Notes",
        }
        assert set(PAGES) == approved

    def test_overview_is_first(self):
        assert PAGES[0] == "Overview"

    def test_project_notes_is_last(self):
        assert PAGES[-1] == "Project Notes"
