"""Chart helpers — Plotly wrappers with centralized theme application.

Every function returns a ``go.Figure`` fully themed for the active mode.
Callers pass ``mode="light"`` or ``mode="dark"`` (from
``st.session_state["theme"]``).  The ``apply_chart_theme`` helper can also be
used to theme any figure built by the caller.

No fabricated data is ever introduced here.  If the caller's DataFrame lacks
the requested column, the function returns an empty figure with an explanatory
annotation so the UI layer can decide what to show.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.theme import LIGHT, DARK, TOKENS, get_plotly_layout


# ── Theme application ──────────────────────────────────────────────────────

def apply_chart_theme(fig: go.Figure, mode: str = "light") -> go.Figure:
    """Apply the active colour tokens to any Plotly figure in-place.

    Sets: plot_bgcolor, paper_bgcolor, font colour/family, axis gridlines,
    tick colours, legend background, hoverlabel colours.

    Returns the same figure object for chaining.
    """
    fig.update_layout(**get_plotly_layout(mode))
    return fig


def _empty_fig(message: str, mode: str = "light") -> go.Figure:
    """Return a blank, themed figure with a centred annotation."""
    t = TOKENS.get(mode, LIGHT)
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font={"size": 14, "color": t["text_secondary"]},
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        **get_plotly_layout(mode),
    )
    return fig


# ── Chart wrappers ─────────────────────────────────────────────────────────

def donut_chart(
    labels: list[str],
    values: list[int | float],
    title: str = "",
    colors: list[str] | None = None,
    mode: str = "light",
) -> go.Figure:
    """At-risk distribution donut chart.

    Parameters
    ----------
    labels:  Slice labels, e.g. ["Not at risk", "At risk"]
    values:  Corresponding counts or fractions.
    title:   Optional chart title.
    colors:  Override slice colours.  Defaults to primary/at_risk tokens.
    mode:    "light" or "dark".
    """
    if not values or sum(values) == 0:
        return _empty_fig("No data to display.", mode)

    t = TOKENS.get(mode, LIGHT)
    default_colors = [t["primary"], t["at_risk"]]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker_colors=colors or default_colors,
            textinfo="label+percent",
            textfont_size=13,
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        )
    )
    if title:
        fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=40 if title else 16, b=24, l=16, r=16),
    )
    return apply_chart_theme(fig, mode)


def histogram_chart(
    df: pd.DataFrame,
    col: str,
    mode: str = "light",
    nbins: int = 30,
) -> go.Figure:
    """Numeric distribution histogram with a box marginal.

    Parameters
    ----------
    df:    DataFrame containing ``col``.
    col:   Column to plot.
    mode:  "light" or "dark".
    nbins: Number of histogram bins.
    """
    if col not in df.columns or df[col].dropna().empty:
        return _empty_fig(f"No data available for '{col}'.", mode)

    t = TOKENS.get(mode, LIGHT)
    fig = px.histogram(
        df,
        x=col,
        nbins=nbins,
        marginal="box",
        color_discrete_sequence=[t["primary"]],
        labels={col: col.replace("_", " ")},
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color=t["border"])
    fig.update_layout(bargap=0.06, margin=dict(t=20, b=20, l=20, r=20))
    return apply_chart_theme(fig, mode)


def box_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mode: str = "light",
    x_labels: dict | None = None,
) -> go.Figure:
    """Box plot of ``y_col`` split by ``x_col`` (typically At_Risk).

    Parameters
    ----------
    df:       DataFrame containing both columns.
    x_col:    Grouping column (x-axis).
    y_col:    Numeric column (y-axis).
    mode:     "light" or "dark".
    x_labels: Optional display-name mapping, e.g. {0: "Not at risk", 1: "At risk"}.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_fig(f"Columns '{x_col}' or '{y_col}' not found.", mode)

    t = TOKENS.get(mode, LIGHT)
    plot_df = df[[x_col, y_col]].dropna().copy()
    if x_labels:
        plot_df[x_col] = plot_df[x_col].map(x_labels).fillna(plot_df[x_col].astype(str))

    fig = px.box(
        plot_df,
        x=x_col,
        y=y_col,
        color=x_col,
        color_discrete_sequence=[t["not_risk"], t["at_risk"]],
        labels={
            x_col: x_col.replace("_", " "),
            y_col: y_col.replace("_", " "),
        },
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
    return apply_chart_theme(fig, mode)


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mode: str = "light",
    title: str = "",
    y_label: str = "",
    color_col: str | None = None,
) -> go.Figure:
    """Vertical bar chart, e.g. group-level at-risk rates.

    Parameters
    ----------
    df:        DataFrame with ``x_col`` and ``y_col``.
    x_col:     Category column (x-axis).
    y_col:     Numeric column (y-axis).
    mode:      "light" or "dark".
    title:     Optional chart title.
    y_label:   Optional y-axis label override.
    color_col: Optional column to map bar colours.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_fig(f"Columns '{x_col}' or '{y_col}' not found.", mode)

    t = TOKENS.get(mode, LIGHT)
    kwargs: dict = dict(
        data_frame=df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[t["primary"]],
        labels={
            x_col: x_col.replace("_", " "),
            y_col: y_label or y_col.replace("_", " "),
        },
    )
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
        kwargs["color_discrete_sequence"] = [t["not_risk"], t["at_risk"]]

    fig = px.bar(**kwargs)
    fig.update_traces(marker_line_width=0)
    if title:
        fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(margin=dict(t=40 if title else 20, b=20, l=20, r=20))
    return apply_chart_theme(fig, mode)


def heatmap_chart(
    corr_df: pd.DataFrame,
    mode: str = "light",
    title: str = "Correlation matrix",
) -> go.Figure:
    """Pearson correlation heatmap.

    Parameters
    ----------
    corr_df: Square correlation DataFrame (output of analytics.correlation_matrix).
    mode:    "light" or "dark".
    title:   Chart title.
    """
    if corr_df.empty:
        return _empty_fig("Not enough numeric columns for a correlation matrix.", mode)

    t = TOKENS.get(mode, LIGHT)
    fig = px.imshow(
        corr_df,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="r"),
        title=title,
    )
    fig.update_traces(
        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
        xgap=1,
        ygap=1,
    )
    fig.update_coloraxes(
        colorbar_tickfont_color=t["text_primary"],
        colorbar_title_font_color=t["text_primary"],
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis_tickangle=-45,
    )
    return apply_chart_theme(fig, mode)


def grouped_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    mode: str = "light",
    title: str = "",
) -> go.Figure:
    """Grouped bar chart for comparing a metric across two categories.

    Parameters
    ----------
    df:        DataFrame with x_col, y_col, group_col.
    x_col:     Primary category axis.
    y_col:     Numeric value.
    group_col: Secondary grouping column.
    mode:      "light" or "dark".
    title:     Optional chart title.
    """
    for col in (x_col, y_col, group_col):
        if col not in df.columns:
            return _empty_fig(f"Column '{col}' not found.", mode)

    t = TOKENS.get(mode, LIGHT)
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=group_col,
        barmode="group",
        color_discrete_sequence=[t["primary"], t["accent"]],
        labels={
            x_col: x_col.replace("_", " "),
            y_col: y_col.replace("_", " "),
        },
    )
    if title:
        fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(margin=dict(t=40 if title else 20, b=20, l=20, r=20))
    return apply_chart_theme(fig, mode)
