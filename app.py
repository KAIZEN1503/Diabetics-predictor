"""
================================================================================
Diabetes Prediction System  -  Premium Redesign
================================================================================
Academic ML project  |  Streamlit 1.30+  |  scikit-learn VotingClassifier
Four tabs: Assess a Patient * Batch Screening * Model Insights * Viva Prep
HARD RULE: inference logic (predict / predict_proba call and DataFrame schema)
is kept identical to the original app.  Only presentation changes.
================================================================================
"""

# ---------------------------------------------------------------------------
# SECTION 0 - IMPORTS
# ---------------------------------------------------------------------------
import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline as SklearnPipeline

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ---------------------------------------------------------------------------
# SECTION 1 - PAGE CONFIG  (must be the VERY FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DiaPredict - Clinical Screening",
    page_icon="stethoscope",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# SECTION 2 - CLINICAL THRESHOLDS  (single source of truth)
# ---------------------------------------------------------------------------
THRESHOLDS: dict = {
    "Pregnancies": {
        "unit": "count", "normal": (0, 6), "elevated": (7, 12),
        "high": (13, 20), "missing_zero": False,
    },
    "Glucose": {
        "unit": "mg/dL", "normal": (70, 140), "elevated": (141, 199),
        "high": (200, 300), "missing_zero": True,
    },
    "BloodPressure": {
        "unit": "mm Hg", "normal": (60, 80), "elevated": (81, 89),
        "high": (90, 160), "missing_zero": True,
    },
    "SkinThickness": {
        "unit": "mm", "normal": (10, 30), "elevated": (31, 50),
        "high": (51, 99), "missing_zero": True,
    },
    "Insulin": {
        "unit": "uU/mL", "normal": (16, 166), "elevated": (167, 300),
        "high": (301, 900), "missing_zero": True,
    },
    "BMI": {
        "unit": "kg/m2", "normal": (18.5, 24.9), "elevated": (25.0, 29.9),
        "high": (30.0, 70.0), "missing_zero": True,
    },
    "DiabetesPedigreeFunction": {
        "unit": "score", "normal": (0.05, 0.40), "elevated": (0.41, 0.80),
        "high": (0.81, 2.50), "missing_zero": False,
    },
    "Age": {
        "unit": "years", "normal": (21, 40), "elevated": (41, 60),
        "high": (61, 120), "missing_zero": False,
    },
}

PRESETS: dict = {
    "Healthy Adult": {
        "Pregnancies": 1, "Glucose": 90.0, "BloodPressure": 72.0,
        "SkinThickness": 22.0, "Insulin": 85.0, "BMI": 22.5,
        "DiabetesPedigreeFunction": 0.18, "Age": 28,
    },
    "Borderline": {
        "Pregnancies": 2, "Glucose": 145.0, "BloodPressure": 84.0,
        "SkinThickness": 32.0, "Insulin": 200.0, "BMI": 29.5,
        "DiabetesPedigreeFunction": 0.52, "Age": 45,
    },
    "High Risk": {
        "Pregnancies": 4, "Glucose": 185.0, "BloodPressure": 92.0,
        "SkinThickness": 42.0, "Insulin": 350.0, "BMI": 36.2,
        "DiabetesPedigreeFunction": 0.95, "Age": 52,
    },
    "Incomplete Report": {
        "Pregnancies": 2, "Glucose": 130.0, "BloodPressure": 76.0,
        "SkinThickness": 0.0, "Insulin": 0.0, "BMI": 27.0,
        "DiabetesPedigreeFunction": 0.38, "Age": 38,
    },
}

PRESET_EMOJI = {
    "Healthy Adult": "Healthy Adult",
    "Borderline": "Borderline",
    "High Risk": "High Risk",
    "Incomplete Report": "Incomplete Report",
}

REQUIRED_BATCH_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Feature order after ColumnTransformer (imputed cols first, then passthrough)
TRANSFORMED_FEATURE_ORDER = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    "Pregnancies", "DiabetesPedigreeFunction", "Age",
]


# ---------------------------------------------------------------------------
# SECTION 3 - GLOBAL CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:#F1F5F9; --surface:#FFFFFF; --border:#E2E8F0;
  --text:#0F172A; --muted:#64748B;
  --blue:#2563EB; --blue-lt:#EFF6FF; --blue-md:#DBEAFE;
  --teal:#059669; --teal-lt:#ECFDF5;
  --amber:#D97706; --amber-lt:#FFFBEB;
  --red:#DC2626; --red-lt:#FEF2F2;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 12px rgba(0,0,0,.08),0 2px 4px rgba(0,0,0,.04);
  --r:14px; --r-sm:8px;
}

*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;color:var(--text);}

#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.4rem!important;padding-bottom:2rem!important;}
section[data-testid="stSidebar"]{display:none;}

/* Tabs */
div[data-testid="stTabs"]>div:first-child{
  background:var(--surface);border-radius:var(--r);padding:4px 6px;
  border:1px solid var(--border);box-shadow:var(--shadow);margin-bottom:1.2rem;
}
button[data-baseweb="tab"]{
  font-size:14px!important;font-weight:500!important;
  border-radius:10px!important;padding:7px 18px!important;
  color:var(--muted)!important;transition:background .15s,color .15s!important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background:var(--blue-lt)!important;color:var(--blue)!important;font-weight:600!important;
}
div[data-testid="stTabsListBorder"]{display:none!important;}

/* Cards */
.dp-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:20px 22px;margin-bottom:16px;
  box-shadow:var(--shadow);
}
.dp-card-title{
  font-size:11px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);margin-bottom:12px;
}

/* Hero */
.dp-hero{
  background:linear-gradient(135deg,#0F172A 0%,#1E3A8A 52%,#2563EB 100%);
  border-radius:var(--r);padding:28px 32px 24px;margin-bottom:16px;
  box-shadow:var(--shadow-md);
}
.dp-hero h1{color:#fff;font-size:26px;font-weight:700;margin:0 0 5px;}
.dp-hero p{color:#CBD5E1;font-size:14px;margin:0 0 14px;line-height:1.6;}
.dp-badges{display:flex;gap:7px;flex-wrap:wrap;}
.dp-badge{
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);
  color:#E0E7FF;font-size:10px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;padding:3px 11px;border-radius:999px;
}

/* Disclaimer */
.dp-disc{
  background:var(--amber-lt);border:1px solid #FDE68A;
  border-radius:var(--r-sm);padding:9px 15px;font-size:12px;
  color:#92400E;margin-bottom:16px;
}

/* Status chips */
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;margin-left:4px;}
.chip-normal{background:#ECFDF5;color:#065F46;border:1px solid #6EE7B7;}
.chip-elevated{background:#FFFBEB;color:#92400E;border:1px solid #FCD34D;}
.chip-high{background:#FEF2F2;color:#991B1B;border:1px solid #FCA5A5;}
.chip-missing{background:#F8FAFC;color:#64748B;border:1px solid #CBD5E1;}

/* KPI */
.dp-kpi{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:18px 14px;text-align:center;box-shadow:var(--shadow);
}
.dp-kpi-lbl{font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:5px;}
.dp-kpi-val{font-size:30px;font-weight:700;color:var(--blue);}

/* Verdict */
.dp-verdict{border-radius:var(--r);padding:20px 22px;margin-bottom:14px;animation:fadeIn .35s ease;}
.dp-v-safe{background:var(--teal-lt);border:2px solid #6EE7B7;}
.dp-v-mod{background:var(--amber-lt);border:2px solid #FCD34D;}
.dp-v-risk{background:var(--red-lt);border:2px solid #FCA5A5;}
.dp-verdict .prob{font-size:46px;font-weight:700;line-height:1.1;}
.dp-verdict h2{margin:5px 0 3px;font-size:20px;}
.dp-verdict p{margin:0;font-size:13px;line-height:1.6;color:var(--muted);}

/* Pipeline diagram */
.dp-pipe{display:flex;align-items:center;overflow-x:auto;padding:6px 0;gap:0;}
.dp-pipe-box{
  background:var(--blue-lt);border:1.5px solid var(--blue-md);color:var(--blue);
  border-radius:var(--r-sm);padding:9px 14px;font-size:12px;font-weight:600;
  white-space:nowrap;text-align:center;
}
.dp-pipe-arr{color:var(--muted);font-size:16px;padding:0 5px;flex-shrink:0;}

/* Empty state */
.dp-empty{text-align:center;padding:44px 20px;color:var(--muted);font-size:14px;}
.dp-empty span{font-size:44px;display:block;margin-bottom:10px;}

@keyframes fadeIn{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}

@media(max-width:768px){
  .dp-hero{padding:18px 16px;}
  .dp-hero h1{font-size:21px;}
  .dp-kpi-val{font-size:24px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECTION 4 - HELPERS: Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(model_path: str = "model.pkl"):
    """Load the fitted scikit-learn Pipeline. Returns None on failure."""
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Failed to load model.pkl: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_metrics(metrics_path: str = "metrics.json"):
    """Load evaluation metrics JSON. Returns None if unavailable."""
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


pipeline = load_pipeline()
metrics_data = load_metrics()


# ---------------------------------------------------------------------------
# SECTION 5 - HELPERS: Inference  (logic unchanged from original app)
# ---------------------------------------------------------------------------

def build_input_df(
    pregnancies: int, glucose: float, blood_pressure: float,
    skin_thickness: float, insulin: float, bmi: float,
    pedigree: float, age: int,
) -> pd.DataFrame:
    """Single-row DataFrame matching the exact training schema."""
    return pd.DataFrame([{
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(pedigree),
        "Age": int(age),
    }])


def run_prediction(input_df: pd.DataFrame):
    """Return (pred_class, prob_non_diabetic%, prob_diabetic%)."""
    pred = pipeline.predict(input_df)[0]
    probs = pipeline.predict_proba(input_df)[0]
    return int(pred), float(probs[0]) * 100, float(probs[1]) * 100


def get_individual_votes(input_df: pd.DataFrame):
    """Per-estimator probabilities via preprocessor+scaler transform.

    Returns dict {name: (p_no%, p_yes%)} or None if structure differs.
    """
    try:
        pre = SklearnPipeline([
            ("preprocessor", pipeline.named_steps["preprocessor"]),
            ("scaler",        pipeline.named_steps["scaler"]),
        ])
        xt = pre.transform(input_df)
        vc = pipeline.named_steps["classifier"]
        return {
            name: (float(est.predict_proba(xt)[0][0]) * 100,
                   float(est.predict_proba(xt)[0][1]) * 100)
            for name, est in vc.named_estimators_.items()
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SECTION 6 - HELPERS: Status chips and validation
# ---------------------------------------------------------------------------

def status_chip(feature: str, value: float) -> str:
    """HTML chip based on clinical thresholds."""
    cfg = THRESHOLDS[feature]
    if cfg["missing_zero"] and value == 0:
        return '<span class="chip chip-missing">Not measured - will be estimated</span>'
    lo_n, hi_n = cfg["normal"]
    lo_e, hi_e = cfg["elevated"]
    if lo_n <= value <= hi_n:
        return '<span class="chip chip-normal">Normal</span>'
    if lo_e <= value <= hi_e:
        return '<span class="chip chip-elevated">Elevated</span>'
    return '<span class="chip chip-high">High</span>'


def risk_label(prob_diabetic: float):
    """Return (label, css_variant) for a given diabetic probability %."""
    if prob_diabetic < 30:
        return "Low Risk", "dp-v-safe"
    if prob_diabetic < 60:
        return "Moderate Risk", "dp-v-mod"
    return "High Risk", "dp-v-risk"


def key_factors(input_df: pd.DataFrame):
    """Top-3 most concerning markers with plain-English messages."""
    row = input_df.iloc[0]
    concerns = []
    for feat, cfg in THRESHOLDS.items():
        val = float(row[feat])
        if cfg["missing_zero"] and val == 0:
            continue
        lo_n, hi_n = cfg["normal"]
        lo_e, hi_e = cfg["elevated"]
        lo_h, hi_h = cfg["high"]
        if lo_h <= val <= hi_h:
            concerns.append((feat, val, 2, cfg["unit"], hi_n))
        elif lo_e <= val <= hi_e:
            concerns.append((feat, val, 1, cfg["unit"], hi_n))
    concerns.sort(key=lambda x: x[2], reverse=True)
    out = []
    for feat, val, sev, unit, threshold in concerns[:3]:
        lbl = feat.replace("DiabetesPedigreeFunction", "Pedigree Score")
        word = "well above" if sev == 2 else "above"
        out.append((lbl, f"{lbl} {val:.1f} {unit} is {word} the normal upper limit of {threshold:.1f} {unit}."))
    return out


def imputed_fields(input_df: pd.DataFrame):
    """Fields that were 0 (missing) and will be median-imputed."""
    row = input_df.iloc[0]
    return [f for f in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            if float(row[f]) == 0]


# ---------------------------------------------------------------------------
# SECTION 7 - HELPERS: Charts
# ---------------------------------------------------------------------------

def gauge_chart(prob: float) -> None:
    """Semicircular risk gauge (Plotly) or progress bar fallback."""
    if not PLOTLY_OK:
        st.progress(int(prob))
        st.metric("Diabetic Probability", f"{prob:.1f}%")
        return

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 40, "color": "#0F172A", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#94A3B8", "tickfont": {"size": 10, "color": "#64748B"}},
            "bar": {"color": "#2563EB", "thickness": 0.26},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "#D1FAE5"},
                {"range": [30, 60],  "color": "#FEF3C7"},
                {"range": [60, 100], "color": "#FEE2E2"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 3},
                          "thickness": 0.8, "value": prob},
            "shape": "angular",
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=230, margin={"t": 10, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def ensemble_votes_chart(votes: dict) -> None:
    """Stacked horizontal bar for each estimator's non-diabetic / diabetic split."""
    if not PLOTLY_OK:
        for name, (p0, p1) in votes.items():
            st.write(f"**{name.replace('_', ' ').title()}**: {p1:.1f}% diabetic")
        return

    names = [n.replace("_", " ").title() for n in votes]
    p_no   = [v[0] for v in votes.values()]
    p_yes  = [v[1] for v in votes.values()]

    fig = go.Figure()
    fig.add_bar(name="Non-Diabetic %", x=p_no,  y=names, orientation="h",
                marker_color="#D1FAE5", marker_line_color="#059669", marker_line_width=1.2)
    fig.add_bar(name="Diabetic %",     x=p_yes, y=names, orientation="h",
                marker_color="#FEE2E2", marker_line_color="#DC2626", marker_line_width=1.2)
    fig.update_layout(
        barmode="stack", height=165,
        margin={"t": 6, "b": 30, "l": 8, "r": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.35, "font": {"size": 11}},
        xaxis={"range": [0, 100], "ticksuffix": "%", "gridcolor": "#F1F5F9"},
        yaxis={"tickfont": {"size": 12}},
        font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def biomarker_bars(input_df: pd.DataFrame) -> None:
    """Marker dots vs. green normal-range bands for all 8 features."""
    if not PLOTLY_OK:
        st.dataframe(input_df)
        return

    row = input_df.iloc[0]
    feats   = list(THRESHOLDS.keys())
    labels  = [f.replace("DiabetesPedigreeFunction", "Pedigree") for f in feats]
    vals    = [float(row[f]) for f in feats]
    max_v   = {f: THRESHOLDS[f]["high"][1] for f in feats}
    pct_val = [min(100, vals[i] / max_v[f] * 100) if max_v[f] else 0
               for i, f in enumerate(feats)]
    pct_lo  = [min(100, THRESHOLDS[f]["normal"][0] / max_v[f] * 100) for f in feats]
    pct_hi  = [min(100, THRESHOLDS[f]["normal"][1] / max_v[f] * 100) for f in feats]

    dot_c = []
    for f, v in zip(feats, vals):
        cfg = THRESHOLDS[f]
        if cfg["missing_zero"] and v == 0:
            dot_c.append("#94A3B8")
        elif cfg["normal"][0] <= v <= cfg["normal"][1]:
            dot_c.append("#059669")
        elif cfg["elevated"][0] <= v <= cfg["elevated"][1]:
            dot_c.append("#D97706")
        else:
            dot_c.append("#DC2626")

    fig = go.Figure()
    for i, (lo, hi) in enumerate(zip(pct_lo, pct_hi)):
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=i - 0.28, y1=i + 0.28,
                      fillcolor="#D1FAE5", opacity=0.5, line={"width": 0})
    fig.add_trace(go.Scatter(
        x=pct_val, y=list(range(len(feats))),
        mode="markers+text",
        text=[f"{v:.1f}" for v in vals],
        textposition="middle right",
        textfont={"size": 10, "color": "#0F172A"},
        marker={"size": 12, "color": dot_c, "line": {"color": "white", "width": 2}},
        hovertext=[f"{lbl}: {v:.1f} {THRESHOLDS[f]['unit']}"
                   for lbl, v, f in zip(labels, vals, feats)],
        hoverinfo="text",
    ))
    fig.update_layout(
        height=330, margin={"t": 6, "b": 8, "l": 8, "r": 55},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAFA",
        xaxis={"range": [0, 112], "visible": False},
        yaxis={"tickvals": list(range(len(feats))), "ticktext": labels,
               "tickfont": {"size": 12}},
        showlegend=False, font={"family": "Inter"},
    )
    fig.add_annotation(x=4, y=len(feats) - 0.65,
                       text="Green band = normal range   Dot = patient value",
                       showarrow=False, font={"size": 9, "color": "#64748B"}, xanchor="left")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def cm_chart(cm: dict) -> None:
    """Heat-mapped confusion matrix."""
    if not PLOTLY_OK:
        st.write(cm)
        return
    z = [[cm.get("tn", 0), cm.get("fp", 0)],
         [cm.get("fn", 0), cm.get("tp", 0)]]
    text = [[f"TN<br>{z[0][0]}", f"FP<br>{z[0][1]}"],
            [f"FN<br>{z[1][0]}", f"TP<br>{z[1][1]}"]]
    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        colorscale=[[0, "#EFF6FF"], [0.5, "#BFDBFE"], [1, "#1E3A8A"]],
        showscale=False,
        x=["Pred: Non-Diabetic", "Pred: Diabetic"],
        y=["Actual: Non-Diabetic", "Actual: Diabetic"],
    ))
    fig.update_layout(
        height=215, margin={"t": 8, "b": 8, "l": 8, "r": 8},
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter", "size": 13},
        xaxis={"side": "top"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def algo_bar_chart(md: dict) -> None:
    """Grouped bar: ensemble vs. three base models on all four metrics."""
    if not PLOTLY_OK or "individual_metrics" not in md:
        return
    keys   = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    ind = md["individual_metrics"]
    models = {
        "Logistic Reg.": ind.get("logistic_regression", {}),
        "Decision Tree":  ind.get("decision_tree", {}),
        "Random Forest":  ind.get("random_forest", {}),
        "Ensemble":       {k: md.get(k, 0) for k in keys},
    }
    colors = ["#93C5FD", "#6EE7B7", "#FCD34D", "#2563EB"]
    fig = go.Figure()
    for (name, vals), color in zip(models.items(), colors):
        fig.add_bar(name=name, y=labels,
                    x=[vals.get(k, 0) * 100 for k in keys],
                    orientation="h", marker_color=color)
    fig.update_layout(
        barmode="group", height=270,
        margin={"t": 8, "b": 8, "l": 8, "r": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 100], "ticksuffix": "%", "gridcolor": "#F1F5F9"},
        legend={"orientation": "h", "y": -0.28, "font": {"size": 11}},
        font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def fi_chart() -> None:
    """Random Forest feature importance horizontal bar."""
    try:
        rf = pipeline.named_steps["classifier"].named_estimators_["random_forest"]
        feat_labels = [f.replace("DiabetesPedigreeFunction", "Pedigree")
                       for f in TRANSFORMED_FEATURE_ORDER]
        df_imp = pd.DataFrame({"Feature": feat_labels,
                               "Importance": rf.feature_importances_ * 100})
        df_imp = df_imp.sort_values("Importance", ascending=True)
        if not PLOTLY_OK:
            st.bar_chart(df_imp.set_index("Feature"))
            return
        fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#DBEAFE", "#2563EB"])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            height=270, margin={"t": 8, "b": 8, "l": 8, "r": 8},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"ticksuffix": "%", "gridcolor": "#F1F5F9"},
            yaxis={"title": None}, font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("Feature importances unavailable from this model structure.")


# ---------------------------------------------------------------------------
# SECTION 8 - HELPERS: Report builder
# ---------------------------------------------------------------------------

def build_report(input_df, pred_class, p_yes, p_no, rlbl, factors, imputed) -> str:
    """Plain-text diagnostic summary report."""
    row = input_df.iloc[0]
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 58,
        "  DIAPREDICT - DIAGNOSTIC SUMMARY",
        "  (Academic Tool - NOT a medical diagnosis)",
        "=" * 58,
        f"  Generated  : {ts}",
        f"  Model      : VotingClassifier (Soft Voting Ensemble)",
        "=" * 58, "",
        "  PATIENT INPUT VALUES", "  " + "-" * 38,
    ]
    for feat, cfg in THRESHOLDS.items():
        lines.append(f"  {feat:<32}: {row[feat]} {cfg['unit']}")
    lines += [
        "", "  PREDICTION RESULT", "  " + "-" * 38,
        f"  Class      : {'Diabetic' if pred_class == 1 else 'Non-Diabetic'}",
        f"  Risk Level : {rlbl}",
        f"  Diabetic % : {p_yes:.1f}%",
        f"  Non-Diab % : {p_no:.1f}%", "",
    ]
    if factors:
        lines += ["  KEY FACTORS", "  " + "-" * 38]
        for i, (_, msg) in enumerate(factors, 1):
            lines.append(f"  {i}. {msg}")
        lines.append("")
    if imputed:
        lines += [
            "  DATA QUALITY", "  " + "-" * 38,
            f"  Imputed fields: {', '.join(imputed)}", "",
        ]
    lines += [
        "=" * 58,
        "  DISCLAIMER: Academic use only. Not a clinical diagnosis.",
        "=" * 58,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 9 - HERO & DISCLAIMER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="dp-hero">
  <h1>DiaPredict &nbsp;&#x1F4CD;</h1>
  <p>Diabetes risk screening powered by a soft-voting ensemble of Logistic
  Regression, Decision Tree &amp; Random Forest trained on the NIDDK Pima
  Indians dataset.</p>
  <div class="dp-badges">
    <span class="dp-badge">Soft-Voting Ensemble</span>
    <span class="dp-badge">Leak-free Pipeline</span>
    <span class="dp-badge">Pima / NIDDK Data</span>
    <span class="dp-badge">Academic Project</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dp-disc">
  &#x26A0;&#xFE0F; &nbsp;<strong>Educational tool only.</strong>
  This system does not provide a medical diagnosis. Always consult a qualified
  clinician for healthcare decisions.
</div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.markdown("""
    <div class="dp-card" style="border-left:4px solid #DC2626;">
      <div class="dp-card-title" style="color:#DC2626;">Model Not Found</div>
      <p>The file <code>model.pkl</code> is missing. Run:</p>
      <pre style="background:#F8FAFC;padding:10px;border-radius:8px;font-size:13px;">python train_model.py</pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# SECTION 10 - TABS
# ---------------------------------------------------------------------------
tab_assess, tab_batch, tab_insights = st.tabs([
    "Assess a Patient",
    "Batch Screening",
    "Model Insights",
])


# ===========================================================================
# TAB 1 - ASSESS A PATIENT
# ===========================================================================
with tab_assess:

    # Session-state defaults
    defaults = PRESETS["Healthy Adult"]
    for k, v in defaults.items():
        if f"field_{k}" not in st.session_state:
            st.session_state[f"field_{k}"] = v
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "gender" not in st.session_state:
        st.session_state["gender"] = "Female"

    # Two-column layout
    col_in, col_res = st.columns([4, 6], gap="large")

    with col_in:
        # Demographics
        st.markdown('<div class="dp-card-title">Demographics</div>', unsafe_allow_html=True)
        gender = st.radio("Biological Sex", ["Female", "Male"],
                          index=0 if st.session_state["gender"] == "Female" else 1,
                          horizontal=True,
                          help="Male locks Pregnancies to 0.",
                          key="gender_radio")
        st.session_state["gender"] = gender
        if gender == "Male":
            st.session_state["field_Pregnancies"] = 0
            st.caption("Training data is female-only (Pima dataset). Male predictions are academic approximations.")

        def labelled_input(feature, label, unit, step, fmt, tip,
                           min_val=None, max_val=None, disabled=False):
            """number_input bound to session_state, with a live status chip."""
            cfg = THRESHOLDS[feature]
            min_v = float(min_val) if min_val is not None else 0.0
            max_v = float(max_val) if max_val is not None else float(cfg["high"][1])
            val = float(st.session_state[f"field_{feature}"])
            val = max(min_v, min(max_v, val))
            new_val = st.number_input(
                f"{label}  ({unit})", min_value=min_v, max_value=max_v,
                value=val, step=float(step), format=fmt,
                disabled=disabled, help=tip, key=f"ni_{feature}",
            )
            st.session_state[f"field_{feature}"] = new_val
            st.markdown(status_chip(feature, new_val), unsafe_allow_html=True)
            return new_val

        # Vitals
        st.markdown('<div class="dp-card-title">Vitals</div>', unsafe_allow_html=True)
        bp_v   = labelled_input("BloodPressure", "Diastolic Blood Pressure", "mm Hg",
                                1.0, "%.0f", "Normal: 60-80 mm Hg.")
        age_v  = labelled_input("Age", "Age", "years", 1, "%d",
                                "Dataset trained on adults >= 21.",
                                min_val=1, max_val=120)
        preg_v = labelled_input("Pregnancies", "Pregnancies", "count", 1, "%d",
                                "Locked to 0 for Male.", min_val=0, max_val=20,
                                disabled=(gender == "Male"))

        # Blood work
        st.markdown('<div class="dp-card-title">Blood Work</div>', unsafe_allow_html=True)
        gluc_v = labelled_input("Glucose", "Plasma Glucose (2h OGTT)", "mg/dL",
                                1.0, "%.0f", "Normal fasting: 70-99. Enter 0 if not tested.")
        ins_v  = labelled_input("Insulin", "2-Hour Serum Insulin", "uU/mL",
                                1.0, "%.0f", "Normal: 16-166. Enter 0 if not tested.")

        # Body & history
        st.markdown('<div class="dp-card-title">Body and History</div>', unsafe_allow_html=True)
        bmi_v  = labelled_input("BMI", "Body Mass Index", "kg/m2",
                                0.1, "%.1f", "18.5-24.9 normal. Enter 0 if unknown.")
        skin_v = labelled_input("SkinThickness", "Triceps Skinfold", "mm",
                                1.0, "%.0f", "Enter 0 if not measured.")
        ped_v  = labelled_input("DiabetesPedigreeFunction", "Diabetes Pedigree Score", "score",
                                0.01, "%.3f", "Genetic family-history risk (0.05-2.5).",
                                min_val=0.05, max_val=2.50)

        bc1, bc2 = st.columns([3, 1])
        with bc1:
            analyze = st.button("Analyze Patient", use_container_width=True,
                                type="primary", key="analyze_btn")
        with bc2:
            reset = st.button("Reset", use_container_width=True, key="reset_btn")

        if reset:
            for k, v in PRESETS["Healthy Adult"].items():
                st.session_state[f"field_{k}"] = v
            st.session_state.update(result=None, gender="Female")
            st.rerun()

    with col_res:
        if analyze:
            idf = build_input_df(
                pregnancies=preg_v if gender == "Female" else 0,
                glucose=gluc_v, blood_pressure=bp_v, skin_thickness=skin_v,
                insulin=ins_v, bmi=bmi_v, pedigree=ped_v, age=int(age_v),
            )
            try:
                pred, p_no, p_yes = run_prediction(idf)
                votes = get_individual_votes(idf)
                st.session_state["result"] = dict(
                    input_df=idf, pred=pred, p_yes=p_yes, p_no=p_no, votes=votes)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        res = st.session_state.get("result")

        if res is None:
            st.markdown("""
            <div class="dp-empty">
              <span>&#x1F3E5;</span>
              Fill in the patient profile and click <strong>Analyze Patient</strong>.
            </div>""", unsafe_allow_html=True)
        else:
            inp   = res["input_df"]
            pred  = res["pred"]
            p_yes = res["p_yes"]
            p_no  = res["p_no"]
            votes = res["votes"]
            rlbl, vcss = risk_label(p_yes)
            factors = key_factors(inp)
            imputed = imputed_fields(inp)

            # Gauge
            st.markdown('<div class="dp-card-title">Diabetic Risk Probability</div>',
                        unsafe_allow_html=True)
            gauge_chart(p_yes)

            # Verdict
            cls_lbl = "Diabetic" if pred == 1 else "Non-Diabetic"
            emoji   = "WARNING" if pred == 1 else "OK"
            vtext   = (f"Profile shows indicators associated with the diabetic class "
                       f"with {p_yes:.1f}% probability." if pred == 1
                       else f"Profile falls in the non-diabetic range "
                            f"with {p_no:.1f}% confidence.")
            st.markdown(f"""
            <div class="dp-verdict {vcss}">
              <div class="dp-card-title">{emoji} - Ensemble Verdict</div>
              <div class="prob">{p_yes:.1f}%</div>
              <h2>{rlbl} &nbsp;&bull;&nbsp; {cls_lbl}</h2>
              <p>{vtext}</p>
            </div>""", unsafe_allow_html=True)

            # Ensemble votes
            if votes:
                st.markdown('<div class="dp-card-title">Ensemble Votes (Individual Models)</div>',
                            unsafe_allow_html=True)
                ensemble_votes_chart(votes)

            # Biomarker chart
            st.markdown('<div class="dp-card-title">Patient Values vs. Normal Range</div>',
                        unsafe_allow_html=True)
            biomarker_bars(inp)

            # Key factors
            if factors:
                st.markdown('<div class="dp-card-title">Key Contributing Factors</div>',
                            unsafe_allow_html=True)
                for _, msg in factors:
                    st.markdown(f'<p style="margin:5px 0;font-size:13px;">- {msg}</p>',
                                unsafe_allow_html=True)

            # Data quality
            if imputed:
                st.markdown(
                    f'<div class="dp-card"><div class="dp-card-title">Data Quality Note</div>'
                    f'<p style="font-size:13px;color:#64748B;margin:0;">'
                    f'Fields entered as 0 (not measured) will be replaced by the training-set median: '
                    f'<strong>{", ".join(imputed)}</strong>.</p></div>',
                    unsafe_allow_html=True)

            # Download report
            report = build_report(inp, pred, p_yes, p_no, rlbl, factors, imputed)
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("Download Diagnostic Report (.txt)",
                               data=report.encode("utf-8"),
                               file_name=f"diapredict_{ts_str}.txt",
                               mime="text/plain",
                               use_container_width=True)


# ===========================================================================
# TAB 2 - BATCH SCREENING
# ===========================================================================
with tab_batch:
    st.markdown('<div class="dp-card-title">Batch Screening - Upload CSV</div>',
                unsafe_allow_html=True)
    st.write("Upload a CSV with the 8 required feature columns. The app scores each row and returns "
             "Prediction, DiabetesProbability (%), and RiskLevel columns.")

    sample = pd.DataFrame([PRESETS[k] for k in PRESETS])
    st.download_button("Download Sample Template CSV",
                       data=sample.to_csv(index=False).encode(),
                       file_name="diapredict_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Choose a CSV", type=["csv"])

    if uploaded:
        try:
            df_b = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            df_b = None

        if df_b is not None:
            missing_cols = [c for c in REQUIRED_BATCH_COLS if c not in df_b.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}. Download the template for the correct format.")
            else:
                bad_types = [c for c in REQUIRED_BATCH_COLS
                             if not pd.api.types.is_numeric_dtype(df_b[c])]
                if bad_types:
                    st.error(f"Non-numeric data in: {', '.join(bad_types)}")
                else:
                    preds_l, pyes_l, rlbl_l = [], [], []
                    with st.spinner("Running predictions..."):
                        for _, row in df_b[REQUIRED_BATCH_COLS].iterrows():
                            try:
                                idf = build_input_df(
                                    int(row["Pregnancies"]), float(row["Glucose"]),
                                    float(row["BloodPressure"]), float(row["SkinThickness"]),
                                    float(row["Insulin"]), float(row["BMI"]),
                                    float(row["DiabetesPedigreeFunction"]), int(row["Age"]),
                                )
                                p_cls, _, p_y = run_prediction(idf)
                                rl, _ = risk_label(p_y)
                            except Exception:
                                p_cls, p_y, rl = -1, float("nan"), "Error"
                            preds_l.append(p_cls)
                            pyes_l.append(round(p_y, 2) if not (p_y != p_y) else float("nan"))
                            rlbl_l.append(rl)

                    df_scored = df_b.copy()
                    df_scored["DiabetesProbability (%)"] = pyes_l
                    df_scored["Prediction"] = [
                        "Diabetic" if p == 1 else "Non-Diabetic" if p == 0 else "Error"
                        for p in preds_l
                    ]
                    df_scored["RiskLevel"] = rlbl_l

                    n_tot  = len(df_scored)
                    n_diab = (df_scored["Prediction"] == "Diabetic").sum()
                    n_mod  = (df_scored["RiskLevel"] == "Moderate Risk").sum()
                    n_high = (df_scored["RiskLevel"] == "High Risk").sum()

                    kc1, kc2, kc3, kc4 = st.columns(4)
                    for col, lbl, val, clr in [
                        (kc1, "Total Rows",      n_tot,  "#2563EB"),
                        (kc2, "Diabetic",         n_diab, "#DC2626"),
                        (kc3, "Moderate Risk",    n_mod,  "#D97706"),
                        (kc4, "High Risk",        n_high, "#DC2626"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class="dp-kpi">
                              <div class="dp-kpi-lbl">{lbl}</div>
                              <div class="dp-kpi-val" style="color:{clr};">{val}</div>
                            </div>""", unsafe_allow_html=True)

                    st.dataframe(df_scored, use_container_width=True, height=300)
                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button("Download Scored CSV",
                                       data=df_scored.to_csv(index=False).encode(),
                                       file_name=f"diapredict_scored_{ts_str}.csv",
                                       mime="text/csv",
                                       use_container_width=True)


# ===========================================================================
# TAB 3 - MODEL INSIGHTS
# ===========================================================================
with tab_insights:
    if metrics_data is None:
        st.warning("metrics.json not found. Run `python train_model.py` to generate it.")
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        for col, lbl, key in [
            (kpi1, "Accuracy",  "accuracy"),
            (kpi2, "Precision", "precision"),
            (kpi3, "Recall",    "recall"),
            (kpi4, "F1-Score",  "f1_score"),
        ]:
            with col:
                v = metrics_data.get(key, 0) * 100
                st.markdown(f"""
                <div class="dp-kpi">
                  <div class="dp-kpi-lbl">{lbl}</div>
                  <div class="dp-kpi-val">{v:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ic1, ic2 = st.columns(2)

        with ic1:
            st.markdown('<div class="dp-card-title">Confusion Matrix</div>',
                        unsafe_allow_html=True)
            cm_chart(metrics_data.get("confusion_matrix", {}))
            fn = metrics_data.get("confusion_matrix", {}).get("fn", 0)
            rec = metrics_data.get("recall", 0) * 100
            st.caption(
                f"False Negatives = {fn}: diabetic patients labelled as non-diabetic. "
                f"The model's recall of {rec:.1f}% means ~1 in 2 true diabetics is missed "
                f"in this dataset - a significant limitation to be aware of in a screening context."
            )

        with ic2:
            st.markdown('<div class="dp-card-title">Algorithm Comparison</div>',
                        unsafe_allow_html=True)
            algo_bar_chart(metrics_data)

        ic3, ic4 = st.columns(2)

        with ic3:
            st.markdown('<div class="dp-card-title">Feature Importance (Random Forest)</div>',
                        unsafe_allow_html=True)
            fi_chart()
            st.caption("Importance extracted from the Random Forest estimator inside the fitted VotingClassifier.")

        with ic4:
            st.markdown('<div class="dp-card-title">Pipeline Architecture</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="dp-pipe">
              <div class="dp-pipe-box">Input<br><small>8 features</small></div>
              <span class="dp-pipe-arr">&#x2192;</span>
              <div class="dp-pipe-box">Median<br>Imputation</div>
              <span class="dp-pipe-arr">&#x2192;</span>
              <div class="dp-pipe-box">Standard<br>Scaling</div>
              <span class="dp-pipe-arr">&#x2192;</span>
              <div class="dp-pipe-box">Soft Voting<br><small>LR + DT + RF</small></div>
              <span class="dp-pipe-arr">&#x2192;</span>
              <div class="dp-pipe-box">Probability<br><small>0 - 100%</small></div>
            </div>
            <br>
            <p style="font-size:12px;color:#64748B;line-height:1.6;">
            Imputer and scaler are fitted <em>only</em> on the 80% training split
            (stratified). Frozen transforms are applied at inference - no test
            data statistics ever enter the training process.
            </p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        ts = metrics_data.get("test_set_size", 154)
        st.markdown(f"""
        <div class="dp-card">
          <div class="dp-card-title">Dataset Summary</div>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr><td style="padding:5px 0;color:#64748B;width:160px;">Source</td>
                <td>NIDDK Pima Indians Diabetes Database</td></tr>
            <tr><td style="padding:5px 0;color:#64748B;">Records</td><td>768 patients</td></tr>
            <tr><td style="padding:5px 0;color:#64748B;">Non-Diabetic</td><td>500 (65.1%)</td></tr>
            <tr><td style="padding:5px 0;color:#64748B;">Diabetic</td><td>268 (34.9%)</td></tr>
            <tr><td style="padding:5px 0;color:#64748B;">Train / Test split</td>
                <td>80% / 20% stratified</td></tr>
            <tr><td style="padding:5px 0;color:#64748B;">Test set size</td>
                <td>{ts} samples</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

