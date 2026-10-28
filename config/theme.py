"""Centralized theme tokens and CSS for the Student At-Risk Analytics Dashboard.

Usage:
    from config.theme import apply_theme, get_plotly_layout, LIGHT, DARK
    st.markdown(apply_theme(st.session_state.get("theme", "light")), unsafe_allow_html=True)
    layout = get_plotly_layout(st.session_state.get("theme", "light"))
"""

LIGHT = {
    "header":           "#1D0B59",
    "primary":          "#5B2BA8",
    "accent":           "#9842BD",
    "bg":               "#F2F6FE",
    "surface":          "#FFFFFF",
    "text_primary":     "#202033",
    "text_secondary":   "#64748B",
    "border":           "#DFE3F2",
    "at_risk":          "#C0392B",
    "not_risk":         "#1A7A4A",
    "at_risk_text":     "#FFFFFF",
    "not_risk_text":    "#FFFFFF",
    "input_bg":         "#FFFFFF",
    "input_text":       "#202033",
    "table_header_bg":  "#EEF2FF",
    "table_row_alt":    "#F8F9FE",
    "alert_info_bg":    "#EEF2FF",
    "alert_info_text":  "#202033",
    "alert_warn_bg":    "#FFF8E1",
    "alert_warn_text":  "#202033",
    "alert_err_bg":     "#FEECEC",
    "alert_err_text":   "#202033",
    "focus_ring":       "#5B2BA8",
}

DARK = {
    "header":           "#160B3D",
    "primary":          "#A78BFA",
    "accent":           "#C084FC",
    "bg":               "#0C1220",
    "surface":          "#151E30",
    "text_primary":     "#F4F6FC",
    "text_secondary":   "#AAB6CB",
    "border":           "#2B3850",
    "at_risk":          "#FF6B6B",
    "not_risk":         "#4ADE80",
    "at_risk_text":     "#160B3D",
    "not_risk_text":    "#0C1220",
    "input_bg":         "#1E2A40",
    "input_text":       "#F4F6FC",
    "table_header_bg":  "#1E2A40",
    "table_row_alt":    "#172030",
    "alert_info_bg":    "#1E2D45",
    "alert_info_text":  "#AAB6CB",
    "alert_warn_bg":    "#2D2510",
    "alert_warn_text":  "#F4F6FC",
    "alert_err_bg":     "#2D1515",
    "alert_err_text":   "#F4F6FC",
    "focus_ring":       "#A78BFA",
}

TOKENS: dict[str, dict[str, str]] = {"light": LIGHT, "dark": DARK}


def apply_theme(mode: str = "light") -> str:
    """Return a <style> block string for injection via st.markdown.

    Themes: html/body background, stApp, all headings, paragraphs, captions,
    st.info/warning/error alerts, dataframe/table cells, input/select borders,
    button styles, scrollbars, and :focus-visible rings.
    Does not rely on Streamlit internal class names that may change between versions;
    uses attribute selectors and standard CSS wherever possible.
    """
    t = TOKENS.get(mode, LIGHT)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Manrope:wght@500;600;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {t["bg"]} !important;
    color: {t["text_primary"]} !important;
    font-family: 'DM Sans', sans-serif;
}}
.stApp {{
    background-color: {t["bg"]} !important;
}}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {{
    color: {t["text_primary"]} !important;
    font-family: 'Manrope', sans-serif;
}}
p, li, span, label, caption {{
    color: {t["text_primary"]};
}}
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {t["text_secondary"]} !important;
}}

/* ── Block container ── */
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 3rem;
    max-width: 1450px;
}}

/* ── Sidebar (if used) ── */
[data-testid="stSidebar"] {{
    background-color: {t["header"]} !important;
}}
[data-testid="stSidebar"] * {{
    color: {t["text_primary"]} !important;
}}

/* ── Cards / surfaces ── */
.at-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 14px;
    padding: 18px 20px;
    color: {t["text_primary"]};
}}
.at-card .label {{
    color: {t["text_secondary"]};
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.at-card .value {{
    color: {t["text_primary"]};
    font-family: 'Manrope', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: 4px;
}}
.at-card .note {{
    color: {t["text_secondary"]};
    font-size: 0.78rem;
    margin-top: 4px;
}}

/* ── Hero banner ── */
.at-hero {{
    background: linear-gradient(120deg, {t["header"]} 0%, {t["primary"]} 60%, {t["accent"]} 100%);
    padding: 24px 32px;
    border-radius: 0 0 18px 18px;
    color: #F4F6FC;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.at-hero-left h1 {{
    font-family: 'Manrope', sans-serif;
    color: #F4F6FC !important;
    font-size: 1.9rem;
    margin: 0 0 6px 0;
}}
.at-hero-left .eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #BDB4F0;
    font-weight: 700;
    font-size: 0.72rem;
    margin-bottom: 8px;
}}
.at-hero-left p {{
    color: #D0C8F8;
    margin: 0;
    font-size: 0.98rem;
}}

/* ── Theme toggle button ── */
.at-toggle-btn {{
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 24px;
    color: #F4F6FC;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.2s;
}}
.at-toggle-btn:hover {{
    background: rgba(255,255,255,0.22);
}}

/* ── Navigation tabs ── */
.at-nav {{
    display: flex;
    gap: 4px;
    background: {t["surface"]};
    border-bottom: 1px solid {t["border"]};
    padding: 0 8px;
    margin-bottom: 24px;
}}
.at-nav-tab {{
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 0.88rem;
    color: {t["text_secondary"]};
    cursor: pointer;
    border: none;
    background: transparent;
    transition: color 0.15s, background 0.15s;
}}
.at-nav-tab:hover {{
    color: {t["primary"]};
    background: {t["bg"]};
}}
.at-nav-tab.active {{
    color: {t["primary"]};
    background: {t["bg"]};
    border-bottom: 2px solid {t["primary"]};
}}

/* ── Inputs, selects, text areas ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] div,
textarea {{
    background-color: {t["input_bg"]} !important;
    color: {t["input_text"]} !important;
    border-color: {t["border"]} !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background-color: {t["surface"]} !important;
    border: 1px dashed {t["border"]} !important;
    border-radius: 10px;
}}
[data-testid="stFileUploader"] label {{
    color: {t["text_secondary"]} !important;
}}

/* ── Alerts: use broad selectors that work across Streamlit versions ── */
div[data-testid="stAlert"] {{
    border-radius: 8px;
}}

/* ── Dataframe / tables ── */
[data-testid="stDataFrame"] th {{
    background-color: {t["table_header_bg"]} !important;
    color: {t["text_primary"]} !important;
}}
[data-testid="stDataFrame"] td {{
    color: {t["text_primary"]} !important;
}}
[data-testid="stDataFrame"] tr:nth-child(even) td {{
    background-color: {t["table_row_alt"]} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {t["primary"]};
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{
    opacity: 0.88;
    color: #FFFFFF;
}}

/* ── Download buttons ── */
.stDownloadButton > button {{
    background-color: {t["surface"]};
    color: {t["primary"]};
    border: 1px solid {t["primary"]};
    border-radius: 10px;
    font-weight: 600;
}}

/* ── Divider ── */
hr {{
    border-color: {t["border"]} !important;
}}

/* ── Focus ring (accessibility) ── */
*:focus-visible {{
    outline: 2px solid {t["focus_ring"]} !important;
    outline-offset: 2px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {t["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {t["border"]}; border-radius: 3px; }}
</style>
"""


def get_plotly_layout(mode: str = "light") -> dict:
    """Return a Plotly layout dict that themes chart backgrounds, axes, and hover labels.

    Apply to any Plotly figure:
        fig.update_layout(**get_plotly_layout(mode))
    """
    t = TOKENS.get(mode, LIGHT)
    return {
        "plot_bgcolor":  t["surface"],
        "paper_bgcolor": t["bg"],
        "font": {
            "color":  t["text_primary"],
            "family": "DM Sans, sans-serif",
        },
        "xaxis": {
            "gridcolor":     t["border"],
            "linecolor":     t["border"],
            "tickcolor":     t["border"],
            "tickfont":      {"color": t["text_secondary"]},
            "title_font":    {"color": t["text_primary"]},
            "zerolinecolor": t["border"],
        },
        "yaxis": {
            "gridcolor":     t["border"],
            "linecolor":     t["border"],
            "tickcolor":     t["border"],
            "tickfont":      {"color": t["text_secondary"]},
            "title_font":    {"color": t["text_primary"]},
            "zerolinecolor": t["border"],
        },
        "legend": {
            "bgcolor":     t["surface"],
            "bordercolor": t["border"],
            "font":        {"color": t["text_primary"]},
        },
        "hoverlabel": {
            "bgcolor":    t["surface"],
            "bordercolor": t["border"],
            "font":       {"color": t["text_primary"]},
        },
        "coloraxis_colorbar": {
            "tickfont":   {"color": t["text_primary"]},
            "title_font": {"color": t["text_primary"]},
        },
    }
