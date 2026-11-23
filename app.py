import base64
import html
import os
from pathlib import Path
from urllib.parse import quote

from gmail_api_sender import send_gmail_api_email
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_risk_model.pkl"
TARGET = "At_Risk"
ID_COLS = {"Student_ID", "Name", "Student_Name"}
NEEDS, TRACK = "At risk", "Not at risk"
LIME, CORAL, BLUE, ORANGE, CREAM = (
    "#9ab800",
    "#ff5252",
    "#3b7de0",
    "#d4470f",
    "#f4f1ea",
)
PALETTE = [CORAL, LIME, BLUE, ORANGE, CREAM]

# SVG Icon definitions
ICON_CAP = '<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M32 12L58 24 32 36 6 24z"/><path d="M16 30v12c8 8 24 8 32 0V30"/><path d="M58 24v16"/></svg>'
ICON_BOOK = '<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 14c9-4 17-3 26 3 9-6 17-7 26-3v34c-9-4-17-3-26 3-9-6-17-7-26-3z"/><path d="M32 17v34"/></svg>'

st.set_page_config(
    page_title="Student Progress | Teacher Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()
model = bundle.get("model") if isinstance(bundle, dict) else bundle
FEATURES = (
    bundle.get("feature_columns", [])
    if isinstance(bundle, dict)
    else list(getattr(model, "feature_names_in_", []))
)

TRAINING_DATA_PATH = ROOT / "training_reference.csv"


@st.cache_resource(show_spinner="Preparing student risk models…")
def load_selectable_models():
    """Train Logistic Regression and Random Forest from the bundled reference data."""
    if not TRAINING_DATA_PATH.exists():
        return {}
    reference = pd.read_csv(TRAINING_DATA_PATH)
    if TARGET not in reference.columns:
        return {}
    X_train = reference.drop(columns=[TARGET]).copy()
    y_train = pd.to_numeric(reference[TARGET], errors="coerce").fillna(0).astype(int)
    categorical = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in X_train.columns if c not in categorical]
    models = {}
    for name, classifier in (
        ("Logistic Regression", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)),
    ):
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if name == "Logistic Regression":
            numeric_steps.append(("scale", StandardScaler()))
        prep = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ], remainder="drop")
        pipeline = Pipeline([("preprocess", prep), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        models[name] = {"model": pipeline, "features": X_train.columns.tolist()}
    return models


def predict_for_choice(raw, choice):
    """Return predictions from the selected classifier; Both flags risk if either does."""
    selectable = load_selectable_models()
    if not selectable:
        # Fall back to the bundled model if the reference dataset is unavailable.
        analysed, missing = analyse_class_sheet(raw)
        return analysed, missing

    available = ["Logistic Regression", "Random Forest"] if choice == "Both" else [choice]
    output = raw.copy().reset_index(drop=True)
    missing_all = set()
    predictions = {}
    scores = {}
    for name in available:
        model_info = selectable[name]
        feature_cols = model_info["features"]
        frame = raw.copy().reset_index(drop=True)
        frame.columns = frame.columns.astype(str).str.strip()
        for col in frame.select_dtypes(include=["object", "string"]).columns:
            frame[col] = frame[col].astype("string").str.strip()
        for col in ["Gender", "Stream", "Higher_Education_Interest"]:
            if col in frame.columns:
                frame[col] = frame[col].str.title()
        missing = [c for c in feature_cols if c not in frame.columns]
        missing_all.update(missing)
        for col in missing:
            frame[col] = np.nan
        X = frame[feature_cols].copy()
        usable = X.notna().sum(axis=1) > 0
        pred = np.zeros(len(X), dtype=int)
        pred[usable.to_numpy()] = model_info["model"].predict(X.loc[usable]).astype(int)
        score = np.where(pred == 1, 100.0, 0.0)
        if hasattr(model_info["model"], "predict_proba") and usable.any():
            try:
                probabilities = model_info["model"].predict_proba(X.loc[usable])
                classes = list(model_info["model"].classes_)
                positive_index = classes.index(1) if 1 in classes else len(classes) - 1
                score[usable.to_numpy()] = np.clip(probabilities[:, positive_index] * 100, 0, 100).round(1)
            except Exception:
                pass
        predictions[name] = pred
        scores[name] = score
        if choice == "Both":
            output[f"{name} Status"] = np.where(pred == 1, NEEDS, TRACK)
            output[f"{name} Support Score"] = score
    if choice == "Both":
        # Conservative two-model agreement: flag a student if either model flags risk.
        combined = np.maximum(predictions["Logistic Regression"], predictions["Random Forest"])
        combined_score = np.maximum(scores["Logistic Regression"], scores["Random Forest"])
        output[TARGET] = combined
        output["Support_Score"] = combined_score
        output["Model Agreement"] = np.where(
            predictions["Logistic Regression"] == predictions["Random Forest"], "Agree", "Disagree"
        )
    else:
        output[TARGET] = predictions[choice]
        output["Support_Score"] = scores[choice]
    output["Status"] = np.where(output[TARGET] == 1, NEEDS, TRACK)
    return output, sorted(missing_all)


def send_student_email(recipient, subject, body):
    """Send email through the Gmail API over HTTPS."""

    def setting(key, default=""):
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except Exception:
            return os.getenv(key, default)

    client_id = setting("GMAIL_CLIENT_ID")
    client_secret = setting("GMAIL_CLIENT_SECRET")
    refresh_token = setting("GMAIL_REFRESH_TOKEN")
    sender = setting("EMAIL_FROM")

    if not all([client_id, client_secret, refresh_token, sender]):
        raise RuntimeError(
            "Gmail API configuration is incomplete. Set GMAIL_CLIENT_ID, "
            "GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN, and EMAIL_FROM."
        )

    send_gmail_api_email(
        recipient,
        subject,
        body,
        client_id,
        client_secret,
        refresh_token,
        sender,
    )

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme = st.session_state.theme
if theme == "Light":
    BG, INK, MUTE, CARD, LINE = (
        "#eaf3ff",
        "#0d2745",
        "#3a5576",
        "#ffffff",
        "#bcd6f4",
    )
    APP_BG = "linear-gradient(180deg,#d6e9ff 0%,#eaf3ff 38%,#ffffff 100%)"
    HERO_BG = "linear-gradient(135deg,#ffffff 0%,#e3f0ff 100%)"
    ACC, LIMET, CORALT = "#1559c7", "#17692f", "#b3261e"
    SHADOW = "0 14px 40px rgba(37,99,235,.14)"
    CHART_TEMPLATE = "plotly_white"
    INPUT_BG = "#ffffff"
    INPUT_BORDER = "#bcd6f4"
    PLACEHOLDER_COLOR = "#5a789a"
    NAV_BG = "#ffffff"
    NAV_BORDER = "#bcd6f4"
    DROP_BG, DROP_BG_HOVER = "#ffffff", "#f0f6ff"
    SCHEME = "light"
    BTN, BTN_H = "#1559c7", "#0d439e"
    UP_BG, UP_BORDER, UP_TXT, UP_HOV_BG, UP_HOV_TXT = (
        "#1559c7",
        "#1559c7",
        "#ffffff",
        "#0d439e",
        "#ffffff",
    )
else:
    BG, INK, MUTE, CARD, LINE = (
        "#0b0b0b",
        "#f4f1ea",
        "#a9a9a3",
        "#141414",
        "#262626",
    )
    APP_BG, HERO_BG = BG, CARD
    ACC, LIMET, CORALT = "#9ab800", "#9ab800", "#ff5252"
    SHADOW = "0 12px 40px rgba(0,0,0,.35)"
    CHART_TEMPLATE = "plotly_dark"
    INPUT_BG = "#1e1e1e"
    INPUT_BORDER = "#333333"
    PLACEHOLDER_COLOR = "#888888"
    NAV_BG = "#141414"
    NAV_BORDER = "#2a2a2a"
    DROP_BG, DROP_BG_HOVER = "#141414", "#1a1a1a"
    SCHEME = "dark"
    BTN, BTN_H = "#2b63b8", "#173d7a"
    UP_BG, UP_BORDER, UP_TXT, UP_HOV_BG, UP_HOV_TXT = (
        "#2b63b8",
        "#2b63b8",
        "#ffffff",
        "#173d7a",
        "#ffffff",
    )


@st.cache_data(show_spinner=False)
def img_uri(name):
    for folder in (ROOT / "images", ROOT):
        p = folder / name
        if p.exists():
            mime = (
                "image/jpeg"
                if p.suffix.lower() in (".jpg", ".jpeg")
                else "image/png"
            )
            return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return None


IMG_GROUP, IMG_DESK, IMG_TALK, IMG_BOARD, IMG_CLASS = (
    img_uri(f"study{i}.png") for i in range(1, 6)
)


def img_tag(uri, alt, cls=""):
    return f'<img class="{cls}" src="{uri}" alt="{html.escape(alt)}">' if uri else ""


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;800&family=Instrument+Serif:ital@1&display=swap');
:root{{--bg:{BG};--ink:{INK};--mute:{MUTE};--card:{CARD};--line:{LINE};--blue:#2b63b8;--btn:{BTN};--btn-h:{BTN_H};--lime:#9ab800;--coral:#ff5252;--orange:#d4470f;--green:#1b7d40;--limet:{LIMET};--coralt:{CORALT};--acc:{ACC};--shadow:{SHADOW};--input-bg:{INPUT_BG};--input-border:{INPUT_BORDER};--placeholder:{PLACEHOLDER_COLOR};--nav-bg:{NAV_BG};--nav-border:{NAV_BORDER};--drop-bg:{DROP_BG};--drop-bg-hover:{DROP_BG_HOVER}}}
html{{scroll-behavior:smooth}}
html,body,[class*="css"],.stMarkdown,button,input,select,textarea{{font-family:'Inter Tight',system-ui,sans-serif!important}}
em{{font-family:'Instrument Serif',Georgia,serif!important;font-style:italic;font-weight:400;letter-spacing:-.02em}}
.stApp{{background:{APP_BG};background-attachment:fixed;color:var(--ink);color-scheme:{SCHEME}}}
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"],header[data-testid="stHeader"],#MainMenu,footer{{display:none!important}}
.block-container{{max-width:100%!important;padding:0 clamp(16px,4vw,64px) 4rem!important}}

h1,h2,h3,h4,h5,p,span,label{{color:var(--ink)}}
.stApp,.stApp p,.stApp li,.stApp label,.stApp span,.stApp h1,.stApp h2,.stApp h3,.stApp h4,
[data-testid="stWidgetLabel"] p,[data-testid="stMarkdownContainer"] p,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{{color:var(--ink)}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,.stApp small{{color:var(--mute)!important}}

/* Header bar */
.st-key-navbar {{
    position: sticky; top: 12px; z-index: 100; width: 100%; margin: 12px 0 28px 0; padding: 8px 24px;
    border-radius: 999px; background-color: var(--nav-bg) !important; border: 1px solid var(--nav-border); box-shadow: var(--shadow);
}}
.st-key-navbar [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: 16px !important; align-items: center; }}
.logo {{ font-size: 22px; font-weight: 800; letter-spacing: -.04em; line-height: 1; white-space: nowrap; display: flex; align-items: center; }}
.logo span {{ color: var(--ink); font-weight: 800; }}
.logo em {{ color: var(--acc); font-style: italic; font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 600; font-size: 1.15em; margin-left: 2px; }}
.st-key-nav div[role="radiogroup"] {{ gap: 8px; justify-content: center; width: fit-content; margin: 0 auto; background: transparent; padding: 0; border: none; }}
.st-key-nav div[role="radiogroup"] label {{ padding: 6px 18px; border-radius: 999px; cursor: pointer; margin: 0; transition: all .25s ease; border: 1px solid transparent; }}
.st-key-nav div[role="radiogroup"] label>div:first-child {{ display: none; }}
.st-key-nav div[role="radiogroup"] label p {{ font-size: 0.9rem; font-weight: 600; color: var(--mute); transition: color .25s; white-space: nowrap; }}
.st-key-nav div[role="radiogroup"] label:hover {{ transform: translateY(-1px); }}
.st-key-nav div[role="radiogroup"] label:hover p {{ color: var(--ink); }}
.st-key-nav div[role="radiogroup"] label:has(input:checked) {{ background: var(--card) !important; border-color: var(--line); }}
.st-key-nav div[role="radiogroup"] label:has(input:checked) p {{ color: var(--ink) !important; font-weight: 700; }}

/* Form inputs */
input[type="text"], input, textarea, div[data-baseweb="input"] input {{
    background-color: var(--input-bg) !important; color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
    border-color: var(--input-border) !important; caret-color: var(--ink) !important;
}}
input::placeholder, textarea::placeholder, div[data-baseweb="input"] input::placeholder {{ color: var(--placeholder) !important; -webkit-text-fill-color: var(--placeholder) !important; opacity: 1 !important; }}
div[data-baseweb="base-input"], div[data-baseweb="input"] {{ background-color: var(--input-bg) !important; border: 1px solid var(--input-border) !important; border-radius: 14px !important; }}
div[data-baseweb="select"] > div {{ background-color: var(--input-bg) !important; border: 1px solid var(--input-border) !important; border-radius: 12px !important; }}
div[data-baseweb="select"] * {{ color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }}
div[data-baseweb="select"] svg {{ fill: var(--ink) !important; }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] li {{ background: var(--card) !important; color: var(--ink) !important; }}
div[data-baseweb="popover"] li * {{ color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }}

/* File Dropzone CSS Fixes */
@keyframes pulseGlow {{ 0%, 100% {{ box-shadow: 0 8px 30px rgba(21,89,199,0.12); border-color: var(--acc); }} 50% {{ box-shadow: 0 12px 40px rgba(21,89,199,0.28); border-color: var(--btn); }} }}
[data-testid="stFileUploader"] {{ max-width: 650px !important; margin: 0 auto 24px auto !important; }}
[data-testid="stFileUploader"] section {{ background: var(--card) !important; background-color: var(--card) !important; border-radius: 24px !important; border: 2px dashed var(--acc) !important; }}
div[data-testid="stFileUploaderDropzone"] {{
    position: relative; overflow: hidden; border: none !important; border-radius: 24px;
    background: var(--card) !important; background-color: var(--card) !important;
    box-shadow: var(--shadow); padding: 32px 24px !important;
    transition: all .35s cubic-bezier(0.4, 0, 0.2, 1); animation: pulseGlow 4s infinite ease-in-out;
}}
div[data-testid="stFileUploaderDropzone"]:hover {{ transform: translateY(-3px); background: var(--drop-bg-hover) !important; }}
div[data-testid="stFileUploaderDropzone"]::before {{
    content: ""; display: block; width: 64px; height: 64px; margin: 0 auto 12px auto;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" stroke="%231559c7" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M16 42a12 12 0 01-2-23.8 16 16 0 0131.6-4.2 12 12 0 0110.4 12 12 12 0 01-10 12"/><polyline points="24 32 32 24 40 32"/><line x1="32" y1="24" x2="32" y2="48"/></svg>');
    background-repeat: no-repeat; background-position: center; background-size: contain;
}}
div[data-testid="stFileUploaderDropzone"] span, div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] div, div[data-testid="stFileUploaderDropzone"] p {{ color: var(--ink) !important; font-weight: 600 !important; }}

div[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
[data-testid="stFileUploader"] button {{
    background-color: {UP_BG} !important; background: {UP_BG} !important;
    border: 2px solid {UP_BORDER} !important; border-radius: 999px !important;
    padding: 0.55rem 1.4rem !important; box-shadow: 0 4px 12px rgba(21,89,199,.3) !important;
    transition: transform .2s ease, background-color .2s ease !important;
}}
div[data-testid="stFileUploaderDropzone"] button *,
div[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] *,
div[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] p,
[data-testid="stFileUploader"] button * {{
    color: {UP_TXT} !important; -webkit-text-fill-color: {UP_TXT} !important; font-weight: 800 !important;
}}
div[data-testid="stFileUploaderDropzone"] button:hover,
div[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stFileUploader"] button:hover {{
    background-color: {UP_HOV_BG} !important; background: {UP_HOV_BG} !important; border-color: {UP_HOV_BG} !important; transform: scale(1.04);
}}
div[data-testid="stFileUploaderDropzone"] button:hover *,
div[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover *,
[data-testid="stFileUploader"] button:hover * {{
    color: {UP_HOV_TXT} !important; -webkit-text-fill-color: {UP_HOV_TXT} !important;
}}

[data-testid="stExpander"] {{ border: 1px solid var(--line) !important; border-radius: 18px; background: var(--card); }}
[data-testid="stExpander"] svg {{ fill: var(--ink) !important; color: var(--ink) !important; }}
div[data-testid="stRadio"] label p {{ color: var(--ink); }}
div[data-testid="stAlert"] {{ background: var(--card) !important; border-radius: 20px; border: 1px solid var(--line); }}
div[data-testid="stAlert"] * {{ color: var(--ink) !important; }}
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] {{ color: var(--ink) !important; }}
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * {{ color: var(--ink) !important; }}

@property --v{{syntax:'<integer>';initial-value:0;inherits:false}}
@keyframes up{{from{{opacity:0;transform:translateY(22px)}}}}
@keyframes slide{{from{{transform:translateY(112%)}}}}
@keyframes pop{{from{{transform:scale(.5);opacity:0}}}}
@keyframes cnt{{from{{--v:0}}to{{--v:var(--n)}}}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(255,82,82,.55)}}70%{{box-shadow:0 0 0 22px rgba(255,82,82,0)}}100%{{box-shadow:0 0 0 0 rgba(255,82,82,0)}}}}
@keyframes dot{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.65)}}}}
@keyframes draw{{to{{stroke-dashoffset:0}}}}

.hero{{position:relative;overflow:hidden;border-radius:40px;border:1px solid var(--line);background:{HERO_BG};box-shadow:var(--shadow);padding:clamp(30px,5vw,72px);margin-bottom:26px;min-height:min(56vh,500px);display:flex;flex-direction:column;justify-content:center}}
.hero .eyebrow{{position:relative;z-index:2;font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute);margin-bottom:16px;font-weight:600}}
.hero h1{{position:relative;z-index:2;margin:0;font-size:clamp(46px,8.5vw,132px);line-height:.9;letter-spacing:-.055em;font-weight:800;max-width:10ch;color:var(--ink)}}
.hero h1 em{{color:var(--acc);font-size:1.02em}}.ln{{display:block;overflow:hidden;padding-bottom:.1em}}.ln>span{{display:block;animation:slide 1.1s cubic-bezier(.2,.8,.1,1) both;animation-delay:calc(var(--i,0)*.12s)}}
.hero p.lead{{position:relative;z-index:2;max-width:480px;color:var(--mute);font-size:1.12rem;line-height:1.45;margin:22px 0 0}}

/* Page Title Font Size Increased */
.sec.head {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 24px; }}
.section-title {{ font-size: clamp(2.2rem, 3.8vw, 3rem) !important; font-weight: 800 !important; line-height: 1.1 !important; letter-spacing: -.03em !important; margin: 0 0 6px 0 !important; color: var(--ink); }}
.section-sub {{ font-size: 1.1rem !important; color: var(--mute); margin: 0; }}
.badge {{ flex: 0 0 auto; width: 48px; height: 48px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); display: grid; place-items: center; color: var(--acc); animation: pop .6s .2s cubic-bezier(.3,1.5,.5,1) both; }}
.badge svg {{ width: 26px; height: 26px; }}

/* How it works */
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:20px;margin:24px 0 4px}}
.step{{background:var(--card);border:1px solid var(--line);border-radius:28px;padding:14px 14px 24px;box-shadow:var(--shadow);animation:up .8s backwards;animation-delay:calc(var(--i,0)*.12s);transition:transform .35s cubic-bezier(.3,1.3,.5,1),box-shadow .35s ease}}
.step:hover{{transform:translateY(-8px) rotate(-.4deg);box-shadow:0 22px 50px rgba(37,99,235,.2)}}
.step .pic{{background:#fff;border-radius:20px;height:clamp(240px,24vw,320px);display:grid;place-items:center;overflow:hidden;border:1px solid var(--line);padding:8px}}
.step .pic img{{width:100%;height:100%;object-fit:contain;display:block}}
.step .num{{display:inline-grid;place-items:center;width:32px;height:32px;border-radius:50%;background:var(--acc);color:#fff!important;font-weight:800;font-size:.95rem;margin:16px 0 6px}}
.step h4{{margin:2px 0 4px;font-size:1.25rem;font-weight:800;letter-spacing:-.03em}}.step p{{margin:0;color:var(--mute);font-size:.98rem;line-height:1.45}}

/* Banner and thumbnails */
.banner{{display:flex;flex-wrap:wrap;align-items:center;gap:clamp(14px,2vw,26px);background:var(--card);border:1px solid var(--line);border-radius:28px;padding:14px 28px 14px 14px;box-shadow:var(--shadow);margin:6px 0 18px;animation:up .7s backwards}}
.banner .pic{{flex:0 0 auto;width:clamp(180px,22vw,260px);height:clamp(150px,16vw,200px);background:#fff;border-radius:22px;border:1px solid var(--line);display:grid;place-items:center;overflow:hidden;padding:6px}}
.banner .pic img{{width:100%;height:100%;object-fit:contain}}
.banner div.txt{{font-size:1.1rem;line-height:1.45;flex:1 1 220px}}.banner div.txt b{{font-size:clamp(1.25rem,2vw,1.6rem);letter-spacing:-.02em}}
.thumb{{flex:0 0 auto;width:96px;height:96px;border-radius:50%;object-fit:cover;object-position:center 20%;background:#fff;border:3px solid var(--line)}}
.empty img.big{{display:block;margin:0 auto 18px;max-width:460px;width:75%;border-radius:26px;background:#fff;border:1px solid var(--line);padding:10px}}

/* Animated Hero Card for Student Search */
@keyframes floatAnim {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-8px); }} }}
.know-hero {{
    position: relative; text-align: center; border-radius: 32px; border: 1px solid var(--line);
    background: var(--card); padding: 48px 24px; box-shadow: var(--shadow); margin: 24px 0;
    overflow: hidden; animation: up .8s ease;
}}
.know-hero .animated-pic {{
    width: 220px; height: 180px; margin: 0 auto 16px auto; background: #fff;
    border-radius: 24px; border: 1px solid var(--line); display: grid; place-items: center;
    overflow: hidden; padding: 12px; animation: floatAnim 3.5s ease-in-out infinite;
}}
.know-hero .animated-pic img {{ width: 100%; height: 100%; object-fit: contain; }}
.know-hero h3 {{ font-size: clamp(24px, 3.5vw, 40px); font-weight: 800; margin: 0 0 8px 0; letter-spacing: -.03em; }}
.know-hero h3 em {{ color: var(--acc); font-style: italic; }}
.know-hero p {{ color: var(--mute); font-size: 1.05rem; margin: 0; }}

/* Metric cards */
.metric-card{{border-radius:30px;padding:24px 26px;min-height:178px;display:flex;flex-direction:column;justify-content:space-between;animation:up .8s backwards;animation-delay:calc(var(--i,0)*.08s);height:100%;transition:transform .25s ease;box-shadow:var(--shadow)}}
.metric-card:hover{{transform:translateY(-4px)}}
.metric-label{{font-size:.9rem;font-weight:600;opacity:.95}}.metric-value{{font-size:clamp(40px,5vw,78px);font-weight:800;letter-spacing:-.055em;line-height:.95;margin-top:10px}}.metric-note{{font-size:.86rem;opacity:.95;margin-top:8px}}
.cnt{{--v:0;counter-reset:c var(--v);animation:cnt 1.4s cubic-bezier(.2,.8,.1,1) forwards}}.cnt::after{{content:counter(c)}}
.metric-card.focus{{animation:up .8s backwards,pulse 2.6s ease-out infinite;animation-delay:calc(var(--i,0)*.08s),1.4s}}
.live{{display:inline-flex;align-items:center;gap:8px;font-size:.74rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-top:8px}}.live i{{width:8px;height:8px;border-radius:50%;background:currentColor;animation:dot 1.4s ease-in-out infinite}}
.insight{{display:flex;gap:16px;align-items:center;border:1px solid var(--line);border-left:5px solid var(--coral);border-radius:22px;padding:14px 24px 14px 16px;background:var(--card);box-shadow:var(--shadow);margin:4px 0 20px;font-size:1.08rem;line-height:1.45;animation:up .7s backwards}}.insight .live{{margin:0}}
.insight.good{{border-left-color:var(--lime)}}
.t-bl{{background:var(--blue);color:#fff}}.t-li{{background:var(--lime);color:#0b0b0b}}.t-co{{background:var(--coral);color:#fff}}.t-wh{{background:var(--ink);color:var(--bg)}}.t-dk{{background:var(--card);border:1px solid var(--line);color:var(--ink)}}
.metric-card.t-bl *,.metric-card.t-co *{{color:#fff}}.metric-card.t-li *{{color:#0b0b0b}}.metric-card.t-wh *{{color:var(--bg)}}

.done{{display:flex;align-items:center;gap:18px;margin:26px 0 8px;animation:up .7s both}}.done h3{{margin:0;font-size:clamp(28px,4vw,56px);letter-spacing:-.05em}}
.chk circle,.chk path{{fill:none;stroke:var(--acc);stroke-width:4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:170;stroke-dashoffset:170;animation:draw .9s .1s ease forwards}}.chk path{{stroke-dasharray:50;stroke-dashoffset:50;animation-delay:.6s}}
.ck{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 18px}}.ck span{{border:1px solid var(--line);border-radius:999px;padding:6px 14px;font-size:13px;color:var(--mute);background:var(--card);animation:pop .5s cubic-bezier(.3,1.6,.5,1) both;animation-delay:calc(var(--i,0)*.1s)}}.ck span.ok{{border-color:var(--acc);color:var(--acc);font-weight:600}}
div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"]{{animation:up .9s backwards}}

/* Buttons */
.stButton button{{
    background-color:var(--btn)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;
    border:2px solid var(--btn)!important;border-radius:999px!important;padding:.65rem 1.5rem!important;
    font-weight:700!important;transition:background-color .2s ease,border-color .2s ease,transform .2s ease!important;
}}
.stDownloadButton button,[data-testid="stDownloadButton"] button{{
    background-color:var(--btn)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;
    border:2px solid var(--btn)!important;border-radius:999px!important;padding:.7rem 1.5rem!important;
    min-height:44px!important;font-weight:700!important;font-size:.95rem!important;
    box-shadow:0 4px 12px rgba(36,86,166,.18)!important;
    transition:background-color .2s ease,border-color .2s ease,transform .2s ease,box-shadow .2s ease!important;
}}
.stButton button *, .stDownloadButton button *, [data-testid="stDownloadButton"] button *{{
    color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:700!important;
}}
.stButton button:hover{{background-color:var(--btn-h)!important;border-color:var(--btn-h)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;transform:translateY(-1px)}}
.stButton button:hover *{{color:#fff!important;-webkit-text-fill-color:#fff!important}}
.stDownloadButton button:hover,[data-testid="stDownloadButton"] button:hover{{
    background-color:var(--btn-h)!important;border-color:var(--btn-h)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;
    box-shadow:0 6px 16px rgba(36,86,166,.3)!important;transform:translateY(-1px);
}}

div[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:22px;overflow:hidden;background:var(--card);box-shadow:var(--shadow)}}
.empty{{border:1px dashed var(--acc);border-radius:32px;padding:40px 24px 54px;text-align:center;background:var(--card);box-shadow:var(--shadow);animation:up .7s both}}.empty h3{{font-size:clamp(26px,3.6vw,46px);margin:0 0 8px}}.empty p{{color:var(--mute)}}

/* Footer */
.foot {{ margin-top: 60px; padding: 32px 20px 24px 20px; border-top: 1px solid var(--line); text-align: center; width: 100%; }}
.foot .fl {{ display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 800; letter-spacing: -.04em; color: var(--ink); }}
.foot .fl span {{ color: var(--ink); }}
.foot .fl em {{ color: var(--acc); font-style: italic; font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 600; margin-left: 2px; }}
.foot p.tagline {{ margin: 6px 0 2px 0; color: var(--mute); font-size: 13px; font-weight: 500; }}
.foot p.copy {{ margin: 0; color: var(--mute); font-size: 12px; opacity: 0.7; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.chip{display:inline-block;border-radius:999px;padding:3px 14px;font-weight:800;font-size:.85rem;margin:2px 0 2px 8px;vertical-align:middle}
.chip.need{background:#ff5252;color:#fff}
.chip.ok{background:#9ab800;color:#0b0b0b}
</style>""",
    unsafe_allow_html=True,
)


def style_fig(fig, height=380):
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter Tight, sans-serif", color=INK),
        colorway=PALETTE,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", title_text="", font=dict(color=INK)
        ),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(color=INK)),
        height=height,
    )
    axis_style = dict(
        gridcolor=(
            "rgba(59,125,224,.18)"
            if theme == "Light"
            else "rgba(128,128,128,.28)"
        ),
        zeroline=False,
        tickfont=dict(color=INK),
        title_font=dict(color=INK),
        linecolor=LINE,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def style_table(df):
    """
    Apply the existing table styling only when the dataframe
    is within Pandas Styler's maximum cell limit.
    For larger datasets, return the plain dataframe.
    """
    max_styler_cells = int(pd.get_option("styler.render.max_elements"))

    # Large dataframes should not use Pandas Styler.
    if df.size > max_styler_cells:
        return df

    styler = df.style

    # Preserve the existing Status highlighting.
    if "Status" in df.columns:
        def highlight_status(value):
            if value == NEEDS:
                return (
                    "background-color: #ff5252;"
                    "color: white;"
                    "font-weight: bold;"
                )
            elif value == TRACK:
                return (
                    "background-color: #9ab800;"
                    "color: black;"
                    "font-weight: bold;"
                )
            return ""

        styler = styler.map(highlight_status, subset=["Status"])

    return styler

def table_height(n_rows, cap=620):
    return int(min(cap, 40 + 35 * max(n_rows, 1)))


def metric_card(
    label, value, note="", tone="dk", i=0, n=None, suffix="", focus=False
):
    shown = (
        f'<span class="cnt" style="--n:{int(n)}"></span>{suffix}'
        if n is not None
        else value
    )
    cls = f"metric-card t-{tone}" + (" focus" if focus else "")
    flag = '<div class="live"><i></i>Needs attention</div>' if focus else ""
    st.markdown(
        f'<div class="{cls}" style="--i:{i}"><div'
        f' class="metric-label">{label}</div><div><div'
        f' class="metric-value">{shown}</div>{flag}<div'
        f' class="metric-note">{note}</div></div></div>',
        unsafe_allow_html=True,
    )


def title(text_html, sub="", icon=""):
    badge = f'<div class="badge">{icon}</div>' if icon else ""
    st.markdown(
        '<div class="sec head"><div><div'
        f' class="section-title">{text_html}</div><div'
        f' class="section-sub">{sub}</div></div>{badge}</div>',
        unsafe_allow_html=True,
    )


def block_title(text, sub=""):
    st.markdown(
        f'<div class="h4">{text}</div><div class="h4s">{sub}</div>',
        unsafe_allow_html=True,
    )


def steps_row():
    items = [
        (
            IMG_DESK,
            "Teacher helping a student",
            "Upload your class sheet",
            "Drop in your Excel or CSV file with your students' details.",
        ),
        (
            IMG_BOARD,
            "Teacher at the blackboard",
            "See your class at a glance",
            "Find out how the class is doing and which students are flagged At"
            " risk.",
        ),
        (
            IMG_TALK,
            "Teacher talking with a student",
            "Talk and support",
            "Use the student list to start a friendly check-in.",
        ),
    ]
    cards = "".join(
        f'<div class="step" style="--i:{i}"><div'
        f' class="pic">{img_tag(uri, alt)}</div><div'
        f' class="num">{i+1}</div><h4>{head}</h4><p>{text}</p></div>'
        for i, (uri, alt, head, text) in enumerate(items)
    )
    st.markdown(f'<div class="steps">{cards}</div>', unsafe_allow_html=True)


def clean_inputs(raw):
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    for c in ["Gender", "Stream", "Higher_Education_Interest"]:
        if c in df.columns:
            df[c] = df[c].str.title()
    df = df.drop(columns=[TARGET], errors="ignore")
    ids = df[[c for c in df.columns if c in ID_COLS]].copy()
    missing = [c for c in FEATURES if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    X = df[FEATURES].copy()
    for c in FEATURES:
        if c in {"Gender", "Stream", "Higher_Education_Interest"}:
            X[c] = X[c].astype("string").str.title()
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    usable = X.notna().sum(axis=1) > 0
    return X, ids, missing, usable


def analyse_class_sheet(raw):
    if model is None or not FEATURES:
        raise RuntimeError(
            "Student insights are temporarily unavailable. Please contact your"
            " administrator."
        )
    X, ids, missing, usable = clean_inputs(raw)
    if not usable.any():
        raise ValueError(
            "No usable student details were found. Please check that the sheet"
            " has the expected columns."
        )
    pred = np.zeros(len(X), dtype=int)
    pred[usable.to_numpy()] = np.asarray(model.predict(X.loc[usable])).astype(
        int
    )
    result = raw.copy().reset_index(drop=True)
    result[TARGET] = pred
    result["Status"] = np.where(result[TARGET] == 1, NEEDS, TRACK)
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            classes = list(getattr(model, "classes_", []))
            positive_index = (
                classes.index(1) if 1 in classes else (len(classes) - 1)
            )
            result["Support_Score"] = (
                np.clip(probs[:, positive_index] * 100, 0, 100).round(1)
            )
        except Exception:
            result["Support_Score"] = np.where(
                result[TARGET] == 1, 100.0, 0.0
            )
    else:
        result["Support_Score"] = np.where(result[TARGET] == 1, 100.0, 0.0)
    return result, missing


def need_data():
    st.markdown(
        f'<div class="empty">{img_tag(IMG_GROUP, "Teacher reading with students", "big")}'
        "<h3>No student file yet.</h3>"
        "<p>Upload your class Excel sheet to see how your students are"
        " doing.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.button(
        "Go to Upload ▸",
        on_click=lambda: st.session_state.update(nav="Upload"),
        key="to_upload_" + st.session_state.get("nav", ""),
    )


def student_labels(df):
    s = None
    for col in ("Name", "Student_Name", "Student_ID"):
        if col in df.columns:
            s = df[col].astype(str)
            break
    if s is None:
        s = pd.Series(
            [f"Student {i+1:03d}" for i in range(len(df))], index=df.index
        )
    dup = s.duplicated(keep=False)
    nums = pd.Series(range(1, len(df) + 1), index=df.index).astype(str)
    s = s.where(~dup, s + " (#" + nums + ")")
    return {int(i): v for i, v in s.items()}


def numeric_detail_cols(df):
    return [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in {TARGET, "Support_Score"}
    ]


def render_all_risk_graphs(results):
    """Renders Overall Class Risk Breakdown as side-by-side bar charts (At Risk vs Not At Risk)

    with dynamic scaling, followed by separate Side-by-Side Attendance and
    Assignment Completion (%) graphs.
    """
    att_col = next((c for c in results.columns if "attend" in c.lower()), None)
    ass_col = next(
        (
            c
            for c in results.columns
            if "assign" in c.lower()
            or "score" in c.lower()
            or "grade" in c.lower()
            or "completion" in c.lower()
        ),
        None,
    )

    # 1. OVERALL CLASS RISK BREAKDOWN (DYNAMIC SCALE ALIGNMENT)
    block_title(
        "1. Overall Class Risk Overview",
        "Bar chart view comparing total At Risk students vs Not At Risk"
        " students",
    )
    ov_c1, ov_c2 = st.columns(2)

    total_count = len(results)
    risk_count = int((results["Status"] == NEEDS).sum())
    ok_count = total_count - risk_count

    risk_pct = (risk_count / total_count * 100) if total_count > 0 else 0
    ok_pct = (ok_count / total_count * 100) if total_count > 0 else 0

    # Calculate shared dynamic Y-axis maximum limit across both charts
    max_count = max(risk_count, ok_count, 1)
    y_max = int(max_count * 1.18) + 1

    with ov_c1:
        st.caption("🔴 **Overall: At-Risk Students Count & Share**")
        fig_ov_risk = px.bar(
            x=["At Risk Students"],
            y=[risk_count],
            color_discrete_sequence=[CORAL],
            text=[f"{risk_count} ({risk_pct:.1f}%)"],
            title="Total At Risk Students",
        )
        fig_ov_risk = style_fig(fig_ov_risk, height=300)
        fig_ov_risk.update_yaxes(
            title_text="Number of Students", range=[0, y_max]
        )
        fig_ov_risk.update_xaxes(title_text="")
        st.plotly_chart(
            fig_ov_risk, use_container_width=True, key="ov_risk_bar_chart"
        )

    with ov_c2:
        st.caption("🟢 **Overall: Not At-Risk Students Count & Share**")
        fig_ov_ok = px.bar(
            x=["Not At Risk Students"],
            y=[ok_count],
            color_discrete_sequence=[LIME],
            text=[f"{ok_count} ({ok_pct:.1f}%)"],
            title="Total Not At Risk Students",
        )
        fig_ov_ok = style_fig(fig_ov_ok, height=300)
        fig_ov_ok.update_yaxes(
            title_text="Number of Students", range=[0, y_max]
        )
        fig_ov_ok.update_xaxes(title_text="")
        st.plotly_chart(
            fig_ov_ok, use_container_width=True, key="ov_ok_bar_chart"
        )

    st.write("")

    # 2. SEPARATE METRIC DISTRIBUTIONS
    if att_col or ass_col:
        block_title(
            "2. Separate Metric Distributions",
            "Detailed Attendance and Assignment Completion (%) breakdown per"
            " risk category",
        )

        if att_col:
            st.markdown("#### **Attendance Breakdown**")
            att_c1, att_c2 = st.columns(2)
            with att_c1:
                st.caption("🔴 **Attendance: At Risk Students**")
                fig_att_risk = px.histogram(
                    results[results["Status"] == NEEDS],
                    x=att_col,
                    nbins=12,
                    color_discrete_sequence=[CORAL],
                )
                fig_att_risk = style_fig(fig_att_risk, height=280)
                fig_att_risk.update_xaxes(title_text="Attendance (%)")
                fig_att_risk.update_yaxes(title_text="At Risk Students")
                st.plotly_chart(
                    fig_att_risk, use_container_width=True, key="sep_att_risk"
                )

            with att_c2:
                st.caption("🟢 **Attendance: Not At Risk Students**")
                fig_att_ok = px.histogram(
                    results[results["Status"] == TRACK],
                    x=att_col,
                    nbins=12,
                    color_discrete_sequence=[LIME],
                )
                fig_att_ok = style_fig(fig_att_ok, height=280)
                fig_att_ok.update_xaxes(title_text="Attendance (%)")
                fig_att_ok.update_yaxes(title_text="Not At Risk Students")
                st.plotly_chart(
                    fig_att_ok, use_container_width=True, key="sep_att_ok"
                )

        if ass_col:
            st.markdown("#### **Assignment Completion (%) Breakdown**")
            ass_c1, ass_c2 = st.columns(2)
            with ass_c1:
                st.caption(
                    "🔴 **Assignment Completion (%): At Risk Students**"
                )
                fig_ass_risk = px.histogram(
                    results[results["Status"] == NEEDS],
                    x=ass_col,
                    nbins=12,
                    color_discrete_sequence=[CORAL],
                )
                fig_ass_risk = style_fig(fig_ass_risk, height=280)
                fig_ass_risk.update_xaxes(
                    title_text="Assignment Completion (%)"
                )
                fig_ass_risk.update_yaxes(title_text="At Risk Students")
                st.plotly_chart(
                    fig_ass_risk, use_container_width=True, key="sep_ass_risk"
                )

            with ass_c2:
                st.caption(
                    "🟢 **Assignment Completion (%): Not At Risk Students**"
                )
                fig_ass_ok = px.histogram(
                    results[results["Status"] == TRACK],
                    x=ass_col,
                    nbins=12,
                    color_discrete_sequence=[LIME],
                )
                fig_ass_ok = style_fig(fig_ass_ok, height=280)
                fig_ass_ok.update_xaxes(title_text="Assignment Completion (%)")
                fig_ass_ok.update_yaxes(title_text="Not At Risk Students")
                st.plotly_chart(
                    fig_ass_ok, use_container_width=True, key="sep_ass_ok"
                )


def attendance_assignment_risk_charts(results, row, labels, idx):
    att_col = next((c for c in results.columns if "attend" in c.lower()), None)
    ass_col = next(
        (
            c
            for c in results.columns
            if "assign" in c.lower()
            or "score" in c.lower()
            or "grade" in c.lower()
            or "completion" in c.lower()
        ),
        None,
    )

    if not att_col and not ass_col:
        return

    block_title(
        "Attendance & Assignment Risk Breakdown",
        "Analyze individual performance against class risk thresholds",
    )

    col1, col2 = st.columns(2)

    with col1:
        if att_col:
            val = float(row[att_col]) if pd.notna(row[att_col]) else 0.0
            avg_val = float(results[att_col].mean())
            att_risk_threshold = 75.0
            is_att_risk = val < att_risk_threshold

            att_cls = "need" if is_att_risk else "ok"
            att_lbl = "At risk" if is_att_risk else "Not at risk"
            att_status_tag = f'<span class="chip {att_cls}">{att_lbl}</span>'
            st.markdown(
                f"**Student Attendance:** **{val:.1f}%** (Class Avg: {avg_val:.1f}%) {att_status_tag}",
                unsafe_allow_html=True,
            )

            fig_att = go.Figure()
            fig_att.add_trace(
                go.Bar(
                    x=["This Student", "Class Average"],
                    y=[val, avg_val],
                    marker_color=[CORAL if is_att_risk else LIME, BLUE],
                    text=[f"{val:.1f}%", f"{avg_val:.1f}%"],
                    textposition="auto",
                )
            )
            fig_att.add_hline(
                y=att_risk_threshold,
                line_dash="dash",
                line_color=ORANGE,
                annotation_text="Risk Threshold (60%)",
                annotation_position="top right",
            )
            fig_att = style_fig(fig_att, height=280)
            fig_att.update_yaxes(title_text="Attendance (%)", range=[0, 105])
            st.plotly_chart(fig_att, use_container_width=True, key=f"att_bar_{idx}")

    with col2:
        if ass_col:
            val = float(row[ass_col]) if pd.notna(row[ass_col]) else 0.0
            avg_val = float(results[ass_col].mean())
            ass_risk_threshold = 50.0
            is_ass_risk = val < ass_risk_threshold

            ass_cls = "need" if is_ass_risk else "ok"
            ass_lbl = "At risk" if is_ass_risk else "Not at risk"
            ass_status_tag = f'<span class="chip {ass_cls}">{ass_lbl}</span>'
            st.markdown(
                f"**Assignment Completion:** **{val:.1f}%** (Class Avg: {avg_val:.1f}%) {ass_status_tag}",
                unsafe_allow_html=True,
            )

            fig_ass = go.Figure()
            fig_ass.add_trace(
                go.Bar(
                    x=["This Student", "Class Average"],
                    y=[val, avg_val],
                    marker_color=[CORAL if is_ass_risk else LIME, BLUE],
                    text=[f"{val:.1f}%", f"{avg_val:.1f}%"],
                    textposition="auto",
                )
            )
            fig_ass.add_hline(
                y=ass_risk_threshold,
                line_dash="dash",
                line_color=ORANGE,
                annotation_text="Risk Threshold (70%)",
                annotation_position="top right",
            )
            fig_ass = style_fig(fig_ass, height=280)
            fig_ass.update_yaxes(
                title_text="Assignment Completion (%)", range=[0, 105]
            )
            st.plotly_chart(fig_ass, use_container_width=True, key=f"ass_bar_{idx}")


def profile_dashboard(results, idx, labels, kp="p"):
    row = results.loc[idx]
    label = labels[int(idx)]
    is_need = row["Status"] == NEEDS
    score = float(row["Support_Score"])
    total = len(results)
    avg_score = float(results["Support_Score"].mean())
    rank = int((results["Support_Score"] > score).sum()) + 1
    same_status = int((results["Status"] == row["Status"]).sum())

    chip = (
        f'<span class="chip {"need" if is_need else "ok"}">{NEEDS if is_need else TRACK}</span>'
    )
    thumb = img_tag(
        IMG_TALK if is_need else IMG_GROUP, "Teacher with student", "thumb"
    )
    msg = (
        "A friendly check-in could help. Listen first, then agree on one small"
        " next step together."
        if is_need
        else "Doing well so far. A word of encouragement goes a long way."
    )

    dash_top_c1, dash_top_c2 = st.columns([3, 1], vertical_alignment="center")
    with dash_top_c1:
        st.markdown(
            f'<div class="insight{"" if is_need else " good"}">{thumb}<div>'
            f'<b style="font-size:1.4rem">{html.escape(label)}</b>'
            f" {chip}<br>{msg}</div></div>",
            unsafe_allow_html=True,
        )
    with dash_top_c2:
        report_data = (
            row.drop(labels=[TARGET], errors="ignore").to_frame().reset_index()
        )
        report_data.columns = ["Parameter / Detail", "Value"]
        report_csv = report_data.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            f"📄 Report Card ({html.escape(str(label))})",
            report_csv,
            f"report_card_{label}.csv",
            "text/csv",
            key=f"dl_report_{idx}",
        )

    k = st.columns(4)
    with k[0]:
        metric_card(
            "Support score",
            "",
            "Out of 100. Higher means more support may help.",
            "co" if is_need else "li",
            0,
            n=round(score),
            suffix="/100",
        )
    with k[1]:
        metric_card(
            "Class average",
            f"{avg_score:.1f}",
            "Average support score of the class",
            "bl",
            1,
        )
    with k[2]:
        metric_card(
            "Support rank",
            f"#{rank}",
            f"Out of {total:,} students (1 = highest score)",
            "wh",
            2,
        )
    with k[3]:
        metric_card(
            "Same risk status",
            "",
            f"Students who are also {(NEEDS if is_need else TRACK).lower()}",
            "dk",
            3,
            n=same_status,
        )

    attendance_assignment_risk_charts(results, row, labels, idx)

    cols_num = numeric_detail_cols(results)

    r1, r2 = st.columns([1.2, 1])
    with r1:
        block_title(
            "Student vs class average", "Each detail scaled 0–100 across the class"
        )
        use = cols_num[:8]
        if len(use) >= 3:
            mn, mx = results[use].min(), results[use].max()
            span = (mx - mn).replace(0, np.nan)
            stu = (
                (
                    (
                        row[use]
                        .astype(float)
                        .fillna(results[use].mean())
                        - mn
                    )
                    / span
                    * 100
                )
                .fillna(50)
                .tolist()
            )
            avg_n = (
                ((results[use].mean() - mn) / span * 100).fillna(50).tolist()
            )
            theta = use + [use[0]]
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=stu + [stu[0]],
                    theta=theta,
                    fill="toself",
                    name="This student",
                    line=dict(color=BLUE, width=3),
                    fillcolor="rgba(59,125,224,.28)",
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=avg_n + [avg_n[0]],
                    theta=theta,
                    name="Class average",
                    line=dict(color=ORANGE, width=2, dash="dash"),
                )
            )
            fig = style_fig(fig, 420)
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        range=[0, 100],
                        gridcolor=LINE,
                        linecolor=LINE,
                        tickfont=dict(color=INK),
                    ),
                    angularaxis=dict(
                        gridcolor=LINE, linecolor=LINE, tickfont=dict(color=INK)
                    ),
                ),
                legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{kp}_radar")
        elif use:
            comp = pd.DataFrame({
                "Detail": use * 2,
                "Value": (
                    list(row[use].astype(float)) + list(results[use].mean())
                ),
                "Who": ["This student"] * len(use) + ["Class average"] * len(use),
            })
            fig = px.bar(
                comp,
                x="Detail",
                y="Value",
                color="Who",
                barmode="group",
                color_discrete_sequence=[BLUE, ORANGE],
            )
            st.plotly_chart(
                style_fig(fig, 400), use_container_width=True, key=f"{kp}_bar"
            )
        else:
            st.info("No numeric columns were found to compare.")

    with r2:
        block_title(
            "All details", "Everything in the spreadsheet for this student"
        )
        data = row.drop(labels=[TARGET], errors="ignore")
        det = pd.DataFrame(
            {"Detail": data.index.astype(str), "Value": [str(v) for v in data.values]}
        )
        st.dataframe(
            style_table(det),
            use_container_width=True,
            hide_index=True,
            height=table_height(len(det), 420),
        )

    st.caption(
        "These are gentle signals, not labels. Talk with the student to understand"
        " their situation before taking action."
    )


PAGES = ["Upload", "Overview", "At-risk students", "Student Support", "AI Study Guide"]

if st.session_state.get("pending_redirect"):
    st.session_state["nav"] = st.session_state.pop("pending_redirect")

with st.container(key="navbar"):
    c1, c2, c3 = st.columns([1.5, 5.5, 1.2], vertical_alignment="center")
    with c1:
        st.markdown(
            '<div class="logo"><span>Student<em>Progress</em></span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        page = st.radio(
            "Navigation",
            PAGES,
            key="nav",
            horizontal=True,
            label_visibility="collapsed",
        )
    with c3:
        st.selectbox(
            "Theme", ["Dark", "Light"], key="theme", label_visibility="collapsed"
        )


results = st.session_state.get("prediction_df")

if page == "Upload":
    st.markdown(
        """
<div class="hero"><div class="sun"></div>
<div class="eyebrow">For teachers · Know your students</div>
<h1><span class="ln"><span style="--i:0">Know your</span></span><span class="ln"><span style="--i:1"><em>students,</em></span></span><span class="ln"><span style="--i:2">early.</span></span></h1>
<p class="lead">Upload your class sheet and see at a glance who is doing well and which students are flagged At risk.</p></div>
""",
        unsafe_allow_html=True,
    )
    if model is None:
        st.error(
            "Student insights are temporarily unavailable. Please contact your"
            " administrator."
        )
    uploaded = st.file_uploader(
        "Class file",
        type=["csv", "xlsx", "xls"],
        key="dataset_upload",
        label_visibility="collapsed",
    )
    st.caption(
        "<div style='text-align:center;'>Upload an Excel (.xlsx/.xls) or CSV"
        " file with your students' details.</div>",
        unsafe_allow_html=True,
    )

    if uploaded is not None:
        try:
            raw = (
                pd.read_excel(uploaded)
                if uploaded.name.lower().endswith((".xlsx", ".xls"))
                else pd.read_csv(uploaded)
            )
            # Clean uploaded column names and standardize important columns
            raw.columns = (
                raw.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )
            def normalize_column_name(col):
                return (
                    str(col).replace("\ufeff", "")
                    .strip()
                    .lower()
                    .replace("_", "")
                    .replace(" ", "")
                    .replace("-", "")
                )

            column_aliases = {
                "studentid": "Student_ID",
                "atrisk": "At_Risk"
            }

            raw = raw.rename(
                columns={
                    col: column_aliases[normalize_column_name(col)]
                    for col in raw.columns
                    if normalize_column_name(col) in column_aliases
                }
            )
            if raw.empty:
                st.error("The uploaded spreadsheet is empty.")
            else:
                with st.spinner("Reviewing your students…"):
                    analysed, missing = analyse_class_sheet(raw)
                st.session_state.update(
                    raw_df=raw,
                    prediction_df=analysed,
                    uploaded_name=uploaded.name,
                    missing_features=missing,
                )
                results = analysed
                st.success("Your class file is ready.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

    if results is not None:
        st.markdown(
            f'<div class="done"><svg class="chk" width="64" height="64" viewBox="0'
            f' 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="26"/><path'
            f' d="M20 33l8 8 16-17"/></svg><h3>{len(results):,} students'
            f" reviewed</h3></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ck"><span class="ok" style="--i:0">✓ File read</span><span'
            ' class="ok" style="--i:1">✓ Students reviewed</span><span class="ok"'
            ' style="--i:2">✓ Results ready to download</span><span'
            f' style="--i:3">{len(st.session_state.get("missing_features", []))}'
            " column(s) not found</span></div>",
            unsafe_allow_html=True,
        )
        if st.session_state.get("missing_features"):
            with st.expander("Columns that would improve the results"):
                st.warning(
                    "Some expected columns were not found in your sheet. Results are"
                    " still shown, but adding these columns gives a more complete"
                    " picture."
                )
                st.write(", ".join(st.session_state["missing_features"]))
        st.dataframe(
            style_table(results.head(8)),
            use_container_width=True,
            hide_index=True,
            height=table_height(min(len(results), 8), 360),
        )
        st.download_button(
            "Download results (CSV)",
            results.to_csv(index=False).encode("utf-8-sig"),
            "student_results.csv",
            "text/csv",
        )
        st.button(
            "Continue to Overview ▸",
            on_click=lambda: st.session_state.update(nav="Overview"),
        )
    else:
        steps_row()


elif page == "Overview":
    title(
        "Class <em>overview.</em>",
        "A quick look at how your class is doing and where support may help.",
        ICON_CAP,
    )
    if results is None:
        need_data()
    else:
        total, need = len(results), int((results[TARGET] == 1).sum())
        ok = total - need
        share = need / total * 100 if total else 0
        thumb = img_tag(IMG_GROUP, "Teacher with students", "thumb")

        st.markdown(
            f'<div class="insight">{thumb}<div><b>Your class, your insights.</b>'
            " Explore academic performance, understand class-wide trends, and identify"
            " students who may benefit from additional support. Use these insights to"
            " guide thoughtful, educator-led follow-ups. <em>Student predictions are"
            " review signals, not final decisions.</em></div></div>",
            unsafe_allow_html=True,
        )

        ip = int(share)
        fr = int(round((share - ip) * 10))
        if fr == 10:
            ip, fr = ip + 1, 0

        k = st.columns(4)
        with k[0]:
            metric_card(
                "Total students", "", "In this uploaded file", "bl", 0, n=total
            )
        with k[1]:
            metric_card(
                "At risk",
                "",
                "Consider a supportive check-in",
                "co",
                1,
                n=need,
                focus=need > 0,
            )
        with k[2]:
            metric_card("Not at risk", "", "Doing well so far", "li", 2, n=ok)
        with k[3]:
            metric_card(
                "Need-support share",
                "",
                "Of your students",
                "wh",
                3,
                n=ip,
                suffix=f".{fr}%",
            )

        a, b = st.columns([1, 1.4])
        with a:
            block_title("Class split", "Share of students by status")
            counts = pd.Series({TRACK: ok, NEEDS: need})
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.68,
                color=counts.index,
                color_discrete_map={TRACK: LIME, NEEDS: CORAL},
            )
            fig.update_traces(
                marker=dict(line=dict(width=0)),
                textinfo="percent",
                textfont=dict(color="#111111"),
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with b:
            group_candidates = [
                c
                for c in ["Gender", "Stream", "Higher_Education_Interest"]
                if c in results.columns and results[c].nunique(dropna=True) <= 15
            ]
            block_title(
                "Support needs by group",
                "Share of each group that could use extra support",
            )
            if group_candidates:
                by = st.selectbox(
                    "Group by", group_candidates, label_visibility="collapsed"
                )
                g = (
                    results.groupby(by, dropna=False)[TARGET]
                    .agg(rate="mean", students="size")
                    .reset_index()
                )
                g["At risk (%)"] = (g["rate"] * 100).round(1)
                fig = px.bar(
                    g,
                    x=by,
                    y="At risk (%)",
                    hover_data=["students"],
                    color_discrete_sequence=[CORAL],
                )
                st.plotly_chart(style_fig(fig, 340), use_container_width=True)
            else:
                st.info(
                    "Add Gender, Stream or Higher_Education_Interest to compare"
                    " groups."
                )

        numeric = [
            c
            for c in results.select_dtypes(include=np.number).columns
            if c not in {TARGET, "Support_Score", "Age"}
        ]
        if numeric:
            d1, d2 = st.columns([1.2, 1])
            with d1:
                block_title("Averages by status", "Simple averages, not causes")
                field = st.selectbox(
                    "Compare", numeric, key="feature_compare", label_visibility="collapsed"
                )
                avg = (
                    results.groupby("Status", dropna=False)[field].mean().reset_index()
                )
                fig = px.bar(
                    avg,
                    x="Status",
                    y=field,
                    color="Status",
                    color_discrete_map={NEEDS: CORAL, TRACK: LIME},
                )
                st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            with d2:
                block_title(
                    "Support score spread",
                    "A higher score means more support may help (0 to 100)",
                )
                fig = px.histogram(
                    results,
                    x="Support_Score",
                    color="Status",
                    nbins=20,
                    barmode="overlay",
                    opacity=0.78,
                    color_discrete_map={NEEDS: CORAL, TRACK: LIME},
                )
                st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        st.caption(
            "These are gentle signals, not labels. Talk with each student to"
            " understand their situation before taking action."
        )



elif page == "At-risk students":
    title(
        "Student <em>list & risk analysis.</em>",
        "Analyze attendance and assignment completion, compare risk models, and search individual students.",
        ICON_BOOK,
    )

    if results is None:
        need_data()
    else:
        # -----------------------------------------
        # 1. PREPARE SOURCE DATA
        # -----------------------------------------
        raw_for_models = st.session_state.get("raw_df")
        if raw_for_models is None:
            raw_for_models = results.drop(
                columns=[TARGET, "Status", "Support_Score"],
                errors="ignore",
            )

        raw_for_models = raw_for_models.copy()
        raw_for_models.columns = (
            raw_for_models.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Normalize common Student_ID column-name variations.
        def _normalise_col_name(value):
            return (
                str(value).replace("\ufeff", "").strip().lower()
                .replace("_", "").replace(" ", "").replace("-", "")
            )

        for col in list(raw_for_models.columns):
            if _normalise_col_name(col) == "studentid" and col != "Student_ID":
                raw_for_models = raw_for_models.rename(columns={col: "Student_ID"})
                break

        # -----------------------------------------
        # 2. MODEL SELECTION
        # -----------------------------------------
        st.write("")
        block_title(
            "1. Student Roster",
            "Choose a prediction model, filter by risk category and search for an individual student.",
        )

        model_options = ["Logistic Regression", "Random Forest", "Both"]
        previous_model = st.session_state.get("risk_model_choice", "Random Forest")
        if previous_model not in model_options:
            previous_model = "Random Forest"

        selected_model = st.selectbox(
            "Choose prediction model",
            model_options,
            index=model_options.index(previous_model),
            key="risk_model_choice",
        )

        try:
            selected_results, selected_missing = predict_for_choice(
                raw_for_models, selected_model
            )
            selected_results = selected_results.copy()
            selected_results.columns = (
                selected_results.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )
            st.session_state["prediction_df"] = selected_results
            st.session_state["missing_features"] = selected_missing
        except Exception as exc:
            st.error(f"Unable to run {selected_model}: {exc}")
            selected_results = st.session_state.get("prediction_df")
            if selected_results is None:
                st.stop()

        # Keep Student_ID from the uploaded source data. The model output is
        # expected to preserve source columns; this is a safe fallback if not.
        if "Student_ID" not in selected_results.columns:
            source_id_col = next(
                (c for c in raw_for_models.columns if _normalise_col_name(c) == "studentid"),
                None,
            )
            if source_id_col is not None and len(raw_for_models) == len(selected_results):
                selected_results["Student_ID"] = raw_for_models[source_id_col].to_numpy()

        if "Student_ID" not in selected_results.columns and not any(
            c in selected_results.columns for c in ("Student_Name", "Name")
        ):
            selected_results.insert(
                0,
                "Student_ID",
                [f"Student {i + 1:03d}" for i in range(len(selected_results))],
            )

        if "Status" not in selected_results.columns:
            selected_results["Status"] = np.where(
                pd.to_numeric(selected_results.get(TARGET, 0), errors="coerce").fillna(0).eq(1),
                NEEDS,
                TRACK,
            )

        view = selected_results.copy()

        # Sort Student_ID naturally in ascending order
        if "Student_ID" in view.columns:
            view["Student_ID"] = view["Student_ID"].astype(str).str.strip()

            # Extract the alphabetic prefix and numeric part
            id_parts = view["Student_ID"].str.extract(r"^(.*?)(\d+)$")

            if id_parts.notna().all().all():
                view["_id_prefix"] = id_parts[0].str.casefold()
                view["_id_number"] = pd.to_numeric(id_parts[1])

                view = (
                    view.sort_values(
                        ["_id_prefix", "_id_number"],
                        ascending=True,
                        kind="stable"
                    )
                    .drop(columns=["_id_prefix", "_id_number"])
                    .reset_index(drop=True)
                )
            else:
                view = view.sort_values(
                    "Student_ID",
                    ascending=True,
                    key=lambda col: col.str.casefold(),
                    kind="stable"
                ).reset_index(drop=True)

        # -----------------------------------------
        # 3. CATEGORY FILTER AND STUDENT SEARCH
        # -----------------------------------------
        f1, f2 = st.columns([1.4, 2], vertical_alignment="bottom")
        with f1:
            show = st.radio(
                "Show students",
                ["All students", NEEDS, TRACK],
                horizontal=True,
                key="risk_roster_filter",
            )
        with f2:
            q = st.text_input(
                "Search student",
                placeholder="Search by ID or student name",
                key="risk_student_search",
            )

        graph_results = view.copy()
        if show in (NEEDS, TRACK):
            graph_results = graph_results[graph_results["Status"] == show]

        if q:
            search_mask = graph_results.astype(str).apply(
                lambda col: col.str.contains(q, case=False, na=False)
            ).any(axis=1)
            graph_results = graph_results[search_mask]

        st.caption(
            f"Showing {len(graph_results):,} of {len(view):,} students "
            f"for the selected category and search."
        )

        # st.download_button(
        #     "Download roster",
        #     graph_results.to_csv(index=False).encode("utf-8-sig"),
        #     "student_roster.csv",
        #     "text/csv",
        #     key="download_roster",
        # )

        # -----------------------------------------
        # 4. INDIVIDUAL STUDENT SEARCH / REDIRECT
        # -----------------------------------------
        if q and st.button("Open matching student", key="open_matching_student"):
            if len(graph_results) == 1:
                matched_idx = graph_results.index[0]
                if graph_results.iloc[0]["Status"] == NEEDS:
                    st.session_state["profile_idx"] = int(matched_idx)
                    st.session_state["profile_q"] = q
                    st.session_state["pending_redirect"] = "Student Support"
                    st.rerun()
                else:
                    st.info("This student is currently classified as Not at risk.")
            elif len(graph_results) == 0:
                st.info("No student matches your search.")
            else:
                st.info("Multiple students found. Please refine your search.")

        # -----------------------------------------
        # 5. ATTENDANCE AND ASSIGNMENT GRAPHS
        # -----------------------------------------
        st.write("")
        block_title(
            "2. Student Performance Analysis",
            "Attendance and assignment completion across the selected students.",
        )

        if graph_results.empty:
            st.info("No student records available for the selected filters.")
        else:
            chart_data = graph_results.copy()
            if "Student_ID" not in chart_data.columns:
                chart_data["Student_ID"] = chart_data.index.astype(str)
            chart_data["Student_ID"] = chart_data["Student_ID"].astype(str)

            status_colors = {
                TRACK: "#17365D",  # Dark blue
                NEEDS: "#FF6B6B",  # Coral
            }
            att_col = next(
                (c for c in chart_data.columns if "attend" in c.lower()), None
            )
            ass_col = next(
                (
                    c for c in chart_data.columns
                    if "assign" in c.lower() or "completion" in c.lower()
                ),
                None,
            )

            if att_col:
                fig_att = px.bar(
                    chart_data,
                    x="Student_ID",
                    y=att_col,
                    color="Status",
                    color_discrete_map=status_colors,
                    category_orders={
                    "Student_ID": chart_data["Student_ID"].drop_duplicates().tolist(),
                    },
                    hover_data={"Student_ID": True, att_col: ":.2f", "Status": True},
                    labels={
                        "Student_ID": "Student_ID",
                        att_col: "Attendance (%)",
                        "Status": "Student Category",
                    },
                    title="Attendance Analysis",
                )
                fig_att.add_hline(
                    y=60,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="Risk Threshold: 60%",
                    annotation_position="top right"
                )
                fig_att.update_xaxes(title_text="Student_ID", showticklabels=False)
                fig_att.update_layout(
                    xaxis_title="Student_ID",
                    yaxis_title="Attendance (%)",
                    legend_title_text="Student Category",
                    bargap=0.25,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#333333")
                )
                #fig_att = style_fig(fig_att)
                # Reapply category colors after the shared theme styling.
                for trace in fig_att.data:
                    if trace.name in status_colors:
                        trace.marker.color = status_colors[trace.name]
                st.plotly_chart(fig_att, use_container_width=True, key="attendance_analysis")
            else:
                st.warning("Attendance column not found in the dataset.")

            if ass_col:
                fig_ass = px.bar(
                    chart_data,
                    x="Student_ID",
                    y=ass_col,
                    color="Status",
                    color_discrete_map=status_colors,
                    category_orders={
                        "Student_ID": chart_data["Student_ID"].drop_duplicates().tolist(),
                        "Status": [TRACK, NEEDS],
                    },
                    hover_data={"Student_ID": True, ass_col: ":.2f", "Status": True},
                    labels={
                        "Student_ID": "Student_ID",
                        ass_col: "Assignment Completion (%)",
                        "Status": "Student Category",
                    },
                    title="Assignment Completion Analysis",
                )
                fig_ass.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="Risk Threshold: 70%",
                    annotation_position="top right"
                )
                fig_ass.update_xaxes(title_text="Student_ID", showticklabels=False)
                fig_ass.update_layout(
                    xaxis_title="Student_ID",
                    yaxis_title="Assignment Completion (%)",
                    legend_title_text="Student Category",
                    bargap=0.25,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#333333")
                )
                #fig_ass = style_fig(fig_ass)
                for trace in fig_ass.data:
                    if trace.name in status_colors:
                        trace.marker.color = status_colors[trace.name]
                st.plotly_chart(fig_ass, use_container_width=True, key="assignment_analysis")
            else:
                st.warning("Assignment completion column not found in the dataset.")

        # -----------------------------------------
        # 6. STUDENT DATASET
        # -----------------------------------------
        st.write("")
        block_title(
            "3. Student Dataset",
            "Dataset filtered according to the selected model, student category and search.",
        )

        if graph_results.empty:
            st.info("No student records to display.")
        else:
            dataset_display = graph_results.copy()
            if "Student_ID" not in dataset_display.columns:
                dataset_display.insert(0, "Student_ID", dataset_display.index.astype(str))

            cols = ["Student_ID"] + [
                col for col in dataset_display.columns if col != "Student_ID"
            ]
            dataset_display = dataset_display[cols]

            st.dataframe(
                style_table(dataset_display),
                use_container_width=True,
                hide_index=True,
                height=table_height(len(dataset_display)),
            )
            st.download_button(
                "Download filtered dataset (CSV)",
                dataset_display.to_csv(index=False).encode("utf-8-sig"),
                "filtered_student_dataset.csv",
                "text/csv",
                key="download_filtered_dataset",
            )

        if st.session_state.get("missing_features"):
            st.caption(
                "Some model input columns were missing and were filled by the "
                "model preprocessing pipeline: "
                + ", ".join(st.session_state["missing_features"])
            )


elif page == "Student Support":
    title(
        "Student <em>support.</em>",
        "Review an individual student's progress and prepare a supportive email.",
        ICON_CAP,
    )
    if results is None:
        need_data()
    else:
        labels = student_labels(results)
        st.button(
            "◂ Back to student list",
            on_click=lambda: st.session_state.update(
                pending_redirect="At-risk students"
            ),
            key="back_list",
                        )

        f1, f2 = st.columns([1.2, 2], vertical_alignment="bottom")
        with f1:
            pq = st.text_input(
                "Find a student",
                placeholder="Type a name, ID or any value",
                key="profile_q",
            )

        ids_all = [int(i) for i in results.index]
        if pq:
            text_mask = (
                results.astype(str)
                .apply(lambda col: col.str.contains(pq, case=False, na=False))
                .any(axis=1)
            )
            ids_all = [
                i
                for i in ids_all
                if text_mask.loc[i] or pq.lower() in labels[i].lower()
            ]

        has_searched = bool(pq) or (
            "profile_idx" in st.session_state
            and st.session_state.get("profile_idx") is not None
        )

        if not has_searched and not pq:
            st.markdown(
                '<div class="know-hero"><div'
                f' class="animated-pic">{img_tag(IMG_TALK, "Teacher and student interaction")}</div><h3>Please'
                " Teacher, <em>Know Your Student</em></h3><p>Search or select a"
                " student above to inspect their detailed performance"
                " breakdown.</p></div>",
                unsafe_allow_html=True,
            )

            with f2:
                selected_student = st.selectbox(
                    "Select a student to view dashboard",
                    [None] + ids_all,
                    format_func=lambda i: (
                        "Choose a student..." if i is None else labels[int(i)]
                    ),
                    key="init_profile_select",
                )
                if selected_student is not None:
                    st.session_state["profile_idx"] = selected_student
                    st.rerun()
        else:
            if not ids_all:
                st.info("No student matches your search. Try another name or ID.")
            else:
                if st.session_state.get("profile_idx") not in ids_all:
                    st.session_state["profile_idx"] = ids_all[0]
                with f2:
                    idx = st.selectbox(
                        "Student",
                        ids_all,
                        key="profile_idx",
                        format_func=lambda i: labels[int(i)],
                    )
                profile_dashboard(results, idx, labels, kp="prof")

                selected_row = results.loc[idx]
                if selected_row.get("Status") == NEEDS:
                    st.write("")
                    block_title(
                        "Email student",
                        "Prepare a formal progress update, review it, and confirm before sending."
                    )

                    # Identify the student's progress columns
                    att_col = next(
                        (c for c in results.columns if "attend" in c.lower()),
                        None
                    )

                    ass_col = next(
                        (
                            c for c in results.columns
                            if "assign" in c.lower() or "completion" in c.lower()
                        ),
                        None
                    )

                    if ass_col is None:
                        ass_col = next(
                            (
                                c for c in results.columns
                                if "score" in c.lower() or "grade" in c.lower()
                            ),
                            None
                        )

                    student_name = labels.get(int(idx), "Student")
                    student_id = selected_row.get("Student_ID", "Not available")
                    support_score = selected_row.get("Support_Score", "Not available")
                    status = selected_row.get("Status", "At risk")

                    # Format progress details
                    attendance_text = (
                        f"{float(selected_row[att_col]):.1f}%"
                        if att_col and pd.notna(selected_row[att_col])
                        else "Not available"
                    )

                    assignment_text = (
                        f"{float(selected_row[ass_col]):.1f}%"
                        if ass_col and pd.notna(selected_row[ass_col])
                        else "Not available"
                    )

                    support_text = (
                        f"{float(support_score):.1f}/100"
                        if pd.notna(support_score)
                        else "Not available"
                    )

                    # Default formal email
                    default_subject = (
                        f"Academic Progress Update and Student Support - {student_name}"
                    )

                    default_body = f"""Dear {student_name},

                    I hope this email finds you well.

                    We are writing to bring to your attention the recent academic progress indicators and to offer our support in helping you improve your performance.

                    Please find your current progress summary below:

                    Student ID: {student_id}
                    Current Risk Status: {status}
                    Attendance: {attendance_text}
                    Assignment Completion: {assignment_text}
                    Support Score: {support_text}

                    Based on the current progress assessment, we would like to understand if you are facing any academic difficulties or challenges that may be affecting your performance.

                    We encourage you to focus on improving your attendance, completing pending assignments, and maintaining consistent academic participation.

                    Our objective is to support you and help you achieve your academic goals. Please feel free to share any concerns or difficulties you may be experiencing.

                    Kindly let us know a convenient time to discuss your progress and identify any additional support that may be helpful.

                    We look forward to your response.

                    Best regards,
                    Student Support Team
                    """

                    email_col1, email_col2 = st.columns(2)

                    with email_col1:
                        recipient = st.text_input(
                            "Student email address",
                            key=f"student_email_{idx}",
                            placeholder="student@example.com"
                        )

                    with email_col2:
                        subject_key = f"email_subject_review_{idx}"
                        st.session_state.setdefault(subject_key, default_subject)

                        email_subject = st.text_input(
                            "Email subject",
                            key=subject_key
                        )

                    draft_key = f"email_body_review_{idx}"

                    # Automatically create the default draft when first opening this student
                    st.session_state.setdefault(draft_key, default_body)

                    if st.button("Reset to default draft", key=f"reset_email_{idx}"):
                        st.session_state[draft_key] = default_body
                        st.rerun()

                    edited_body = st.text_area(
                        "Review and edit email before sending",
                        key=draft_key,
                        height=350
                    )

                    confirm_send = st.checkbox(
                        "I have reviewed this email and confirm that it should be sent to the student.",
                        key=f"confirm_send_{idx}"
                    )

                    if st.button(
                        "Send email",
                        key=f"send_email_{idx}",
                        type="primary"
                    ):
                        if not recipient or "@" not in recipient:
                            st.error("Please enter a valid student email address.")

                        elif not confirm_send:
                            st.warning("Please confirm the email after reviewing it.")

                        elif not edited_body.strip():
                            st.warning("Email body cannot be empty.")

                        else:
                            try:
                                send_student_email(
                                    recipient.strip(),
                                    email_subject,
                                    edited_body
                                )
                                st.success("Email sent successfully.")

                            except Exception as exc:
                                st.error(f"Email was not sent: {exc}")

elif page == "AI Study Guide":
    from ai_study_guide.study_guide_view import render_study_guide
    render_study_guide(results, st.session_state.get("raw_df"))

st.markdown(
    """
<div class="foot">
    <div class="fl"><span>Student<em>Progress</em></span></div>
    <p class="tagline">Supporting teachers with student progress insights.</p>
    <p class="copy">© 2026 StudentProgress · Educational prototype</p>
</div>
""",
    unsafe_allow_html=True,
)
