"""
Dashboard Dash — Prediction de Resiliation Telecom
Cours 8INF436 — Forage des Donnees — UQAC Hiver 2026
"""

import json
import sys
import warnings
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

warnings.filterwarnings("ignore")

# Chemins et chargement des artefacts
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.calibration import IsotonicCalibratedModel, PlattCalibratedModel  # noqa requis pour unpickle

scaler         = joblib.load(ROOT / "models" / "scaler.pkl")
scale_cols     = joblib.load(ROOT / "models" / "scale_cols.pkl")
selected_feats = joblib.load(ROOT / "models" / "selected_features.pkl")

with open(ROOT / "models" / "metrics.json") as f:
    metrics_data = json.load(f)

with open(ROOT / "data" / "processed" / "feature_names.json") as f:
    all_features = json.load(f)

MODEL_NAMES  = ["Random Forest", "XGBoost", "MLP"]
MODEL_COLORS = {"Random Forest": "#1565C0", "XGBoost": "#C62828", "MLP": "#2E7D32"}

loaded_models = {}
for name in MODEL_NAMES:
    key = name.lower().replace(" ", "_")
    p = ROOT / "models" / f"{key}.pkl"
    if p.exists():
        loaded_models[name] = joblib.load(p)

# Profil de base : medianes des clients actifs sur le jeu de test.
# Les features non exposees dans le formulaire prennent ainsi des valeurs
# representatives issues de vrais clients, evitant les profils hors-distribution.
from sklearn.model_selection import train_test_split as _tts
_X_proc = pd.read_parquet(ROOT / "data" / "processed" / "X_preprocessed.parquet")
_y_proc = pd.read_parquet(ROOT / "data" / "processed" / "y_preprocessed.parquet").squeeze()
_, _Xtest, _, _ytest = _tts(_X_proc[selected_feats], _y_proc,
                             test_size=0.20, random_state=42, stratify=_y_proc)
feature_defaults = _Xtest[_ytest == 0].median().to_dict()
del _X_proc, _y_proc, _Xtest, _ytest

# Parametres du MinMaxScaler par colonne pour un scaling individuel
_scaler_cols   = list(scaler.feature_names_in_)
_scaler_scale  = {c: float(scaler.scale_[i]) for i, c in enumerate(_scaler_cols)}
_scaler_offset = {c: float(scaler.min_[i])   for i, c in enumerate(_scaler_cols)}

def _scale_val(col: str, raw: float) -> float:
    """Applique le MinMaxScaler a une seule valeur brute."""
    if col in _scaler_scale:
        return raw * _scaler_scale[col] + _scaler_offset[col]
    return raw


# Champs du formulaire : (cle, label, min, max, defaut)
INPUT_FIELDS = [
    ("NB_REENGAGEMENTS",     "Nb reengagements",               0,    5,    0),
    ("JOURS_FIN_ENGAGEMENT", "Jours restants engagement",      0,  761,   90),
    ("ANCIENNETE_MOIS",      "Anciennete (mois)",              0,  240,   29),
    ("DUREE_OFFRE",          "Duree offre actuelle (mois)",    1,   24,    3),
    ("NB_SERVICES",          "Nombre de services",             1,   10,    3),
    ("AGE",                  "Age (annees)",                  18,   90,   36),
    ("VOL_APPELS_M1",        "Volume appels mois dernier",     0,50000,16500),
    ("NB_SMS_M1",            "Nb SMS mois dernier",            0,  500,   31),
]

TOGGLE_FIELDS = [
    ("FLAG_APPELS_VERS_INTERNATIONAL",   "Appels internationaux sortants"),
    ("FLAG_APPELS_DEPUIS_INTERNATIONAL", "Appels internationaux entrants"),
    ("FLAG_APPELS_NUMEROS_SPECIAUX",     "Appels numeros speciaux"),
    ("ENGAGEMENT_EXPIRE",                "Engagement expire"),
    ("FLAG_MIGRATION_HAUSSE",            "Migration montee en gamme"),
    ("FLAG_MIGRATION_BAISSE",            "Migration descente en gamme"),
]

METRIC_LABELS = {
    "Accuracy": "Accuracy", "Precision": "Precision",
    "Recall": "Rappel", "F1-Score": "F1-Score",
    "ROC-AUC": "ROC-AUC", "AP (PR-AUC)": "PR-AUC",
}


def risk_info(prob: float):
    if prob < 0.30:
        return "Faible risque",   "#198754", "Client peu susceptible de resilier."
    elif prob < 0.55:
        return "Risque modere",   "#ffc107", "Profil ambigu, surveillance recommandee."
    elif prob < 0.75:
        return "Risque eleve",    "#dc3545", "Signaux clairs de resiliation imminente."
    else:
        return "Risque critique", "#7B0000", "Client tres probablement sur le point de resilier."


def make_gauge(prob: float, model_name: str, model_color: str) -> go.Figure:
    """Jauge Plotly Indicator affichant la probabilite de resiliation."""
    pct = round(prob * 100, 1)
    risk_label, bar_color, _ = risk_info(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 28, "color": bar_color}},
        title={"text": f"<b>{model_name}</b><br><span style='font-size:12px;color:{bar_color}'>{risk_label}</span>",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#adb5bd",
                     "tickvals": [0, 30, 55, 75, 100],
                     "ticktext": ["0", "30", "55", "75", "100"]},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#d4edda"},
                {"range": [30, 55], "color": "#fff3cd"},
                {"range": [55, 75], "color": "#f8d7da"},
                {"range": [75, 100], "color": "rgba(192,57,43,0.12)"},
            ],
            "threshold": {
                "line": {"color": model_color, "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin={"t": 60, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


def metrics_bar_chart() -> go.Figure:
    metrics_display = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    for model_name, color in MODEL_COLORS.items():
        vals = [metrics_data.get(m, {}).get(model_name, 0) for m in metrics_display]
        fig.add_trace(go.Bar(
            name=model_name, x=[METRIC_LABELS[m] for m in metrics_display], y=vals,
            marker_color=color, opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition="outside", textfont_size=10,
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Metriques d'evaluation — Jeu de test", font_size=14),
        yaxis=dict(range=[0, 1.15], title="Score", gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20), height=380,
    )
    return fig


def radar_chart() -> go.Figure:
    cats = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    for model_name, color in MODEL_COLORS.items():
        vals = [metrics_data.get(c, {}).get(model_name, 0) for c in cats]
        vals_c = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats + [cats[0]], fill="toself",
            name=model_name, line_color=color, opacity=0.5,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=dict(text="Profil de performance", font_size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=370, margin=dict(t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Prediction Resiliation Telecom — 8INF436",
    suppress_callback_exceptions=True,
)

HEADER = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(
                src="/assets/uqac_logo.png", height="36px",
                style={"filter": "brightness(0) invert(1)", "marginRight": "12px"},
            ), width="auto"),
            dbc.Col(dbc.NavbarBrand(
                "Prediction de Resiliation Telecom",
                style={"fontWeight": "700", "fontSize": "1.15rem"},
            ), width="auto"),
        ], align="center", className="g-0 flex-grow-1"),
        html.Span("8INF436 — UQAC Hiver 2026",
                  style={"color": "rgba(255,255,255,.70)", "fontSize": ".85rem",
                         "whiteSpace": "nowrap"}),
    ], fluid=True),
    color="primary", dark=True, className="mb-4 shadow-sm",
)

METRICS_TAB = dbc.Tab(label="Performances des modeles", tab_id="tab-metrics", children=[
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-metrics-bar", figure=metrics_bar_chart(),
                          config={"displayModeBar": False}), md=7),
        dbc.Col(dcc.Graph(id="fig-radar", figure=radar_chart(),
                          config={"displayModeBar": False}), md=5),
    ], className="mt-3"),
    dbc.Row([dbc.Col(html.Div(id="metrics-table-div"))], className="mt-2"),
])

PREDICT_TAB = dbc.Tab(label="Classifier un client", tab_id="tab-predict", children=[
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Attributs du client", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label(label, style={"fontSize": ".82rem", "marginBottom": "2px"}),
                            dcc.Slider(id=f"sl-{key}", min=mn, max=mx, value=default,
                                       step=max(1, (mx - mn) // 100),
                                       tooltip={"placement": "bottom", "always_visible": True},
                                       marks=None),
                            html.Div(className="mb-3"),
                        ], md=6)
                        for key, label, mn, mx, default in INPUT_FIELDS
                    ]),
                    html.Hr(),
                    dbc.Checklist(
                        id="flags",
                        options=[{"label": lbl, "value": key} for key, lbl in TOGGLE_FIELDS],
                        value=[], switch=True,
                    ),
                    html.Hr(),
                    dcc.Upload(
                        id="upload-csv",
                        children=html.Div([
                            html.I(className="bi bi-cloud-upload me-2"),
                            "Glisser-deposer ou ", html.A("parcourir"),
                        ]),
                        style={
                            "width": "100%", "height": "50px", "lineHeight": "50px",
                            "borderWidth": "2px", "borderStyle": "dashed",
                            "borderRadius": "8px", "borderColor": "#adb5bd",
                            "textAlign": "center", "fontSize": ".82rem", "color": "#6c757d",
                        },
                        multiple=False,
                    ),
                    html.Div(className="mt-3"),
                    dbc.Button(
                        [html.I(className="bi bi-lightning-fill me-2"), "Classifier"],
                        id="btn-predict", color="primary", size="lg", className="w-100 fw-semibold",
                    ),
                ]),
            ], className="shadow-sm"),
        ], md=5),
        dbc.Col([
            html.Div(id="predict-info", children=[
                dbc.Alert([html.I(className="bi bi-info-circle me-2"),
                           "Renseignez les attributs puis cliquez sur Classifier."],
                          color="info", className="text-center"),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="gauge-rf",  config={"displayModeBar": False},
                                  style={"display": "none"}), md=4),
                dbc.Col(dcc.Graph(id="gauge-xgb", config={"displayModeBar": False},
                                  style={"display": "none"}), md=4),
                dbc.Col(dcc.Graph(id="gauge-mlp", config={"displayModeBar": False},
                                  style={"display": "none"}), md=4),
            ], className="g-2 mb-2"),
            html.Div(id="predict-consensus"),
        ], md=7),
    ], className="mt-3 g-4"),
])

app.layout = dbc.Container([
    HEADER,
    dbc.Tabs([METRICS_TAB, PREDICT_TAB], id="main-tabs", active_tab="tab-metrics"),
    html.Footer(
        html.Small("Forage des Donnees 8INF436 — UQAC Hiver 2026 — Dataset Telecom Churn",
                   className="text-muted"),
        className="text-center py-4 mt-5",
    ),
], fluid=True)


@app.callback(
    Output("metrics-table-div", "children"),
    Input("main-tabs", "active_tab"),
)
def render_metrics_table(_):
    rows = []
    for mk in METRIC_LABELS:
        cells = [html.Td(METRIC_LABELS[mk], style={"fontWeight": "600"})]
        for mn in loaded_models:
            val = metrics_data.get(mk, {}).get(mn, None)
            cells.append(html.Td(f"{val:.4f}" if isinstance(val, float) else "—",
                                 className="text-center"))
        rows.append(html.Tr(cells))
    header = html.Thead(html.Tr(
        [html.Th("Metrique")] + [html.Th(m, className="text-center") for m in loaded_models]
    ), className="table-dark")
    return dbc.Table([header, html.Tbody(rows)],
                     striped=True, hover=True, responsive=True, className="mt-2 small")


def _build_vector(slider_vals, flags):
    safe = [v if v is not None else d for v, (_, _, _, _, d) in zip(slider_vals, INPUT_FIELDS)]

    # Base = medianes des clients actifs (espace scale).
    # Les features non exposees gardent ainsi des valeurs coherentes avec la population reelle.
    vec = pd.DataFrame([{c: feature_defaults.get(c, 0.0) for c in all_features}])

    for key, _ in TOGGLE_FIELDS:
        if key in vec.columns:
            vec[key] = float(1 if key in (flags or []) else 0)

    raw_slider: dict = {}
    for (k, *_), v in zip(INPUT_FIELDS, safe):
        raw_slider[k] = float(v)
        if k in vec.columns:
            vec[k] = _scale_val(k, float(v))

    # Coherence temporelle : mois M2 a M6 = meme volume que M1 (pas d'historique artificiel)
    scaled_vol = _scale_val("VOL_APPELS_M1", raw_slider.get("VOL_APPELS_M1", 16500))
    scaled_sms = _scale_val("NB_SMS_M1",     raw_slider.get("NB_SMS_M1",      31))
    for col in ["VOL_APPELS_M2", "VOL_APPELS_M3", "VOL_APPELS_M4",
                "VOL_APPELS_M5", "VOL_APPELS_M6", "VOL_APPELS_MOY"]:
        if col in vec.columns:
            vec[col] = scaled_vol
    for col in ["NB_SMS_M2", "NB_SMS_M3", "NB_SMS_M4", "NB_SMS_M5", "NB_SMS_M6", "NB_SMS_MOY"]:
        if col in vec.columns:
            vec[col] = scaled_sms

    # Duree initiale = duree actuelle (coherent pour un client sans re-engagement)
    if "DUREE_OFFRE_INIT" in vec.columns:
        vec["DUREE_OFFRE_INIT"] = _scale_val("DUREE_OFFRE", raw_slider.get("DUREE_OFFRE", 3))

    return vec[selected_feats]


@app.callback(
    Output("predict-info",      "children"),
    Output("gauge-rf",          "figure"),
    Output("gauge-xgb",         "figure"),
    Output("gauge-mlp",         "figure"),
    Output("gauge-rf",          "style"),
    Output("gauge-xgb",         "style"),
    Output("gauge-mlp",         "style"),
    Output("predict-consensus", "children"),
    Input("btn-predict", "n_clicks"),
    State("flags", "value"),
    [State(f"sl-{k}", "value") for k, *_ in INPUT_FIELDS],
    prevent_initial_call=True,
)
def classify(n_clicks, flags, *slider_vals):
    vec = _build_vector(list(slider_vals), flags)

    probs, preds = [], []
    for name in MODEL_NAMES:
        if name not in loaded_models:
            continue
        p = float(loaded_models[name].predict_proba(vec)[0][1])
        probs.append(p)
        preds.append(int(p >= 0.5))

    fig_rf  = make_gauge(probs[0], "Random Forest", MODEL_COLORS["Random Forest"])
    fig_xgb = make_gauge(probs[1], "XGBoost",       MODEL_COLORS["XGBoost"])
    fig_mlp = make_gauge(probs[2], "MLP",            MODEL_COLORS["MLP"])
    gauge_style = {"display": "block"}

    vote   = sum(preds)
    p_mean = float(np.mean(probs))
    p_std  = float(np.std(probs))
    label  = "Resilie" if vote >= 2 else "Actif"
    color  = "danger"  if vote >= 2 else "success"
    icon   = "bi-exclamation-triangle-fill" if vote >= 2 else "bi-check-circle-fill"

    consensus = dbc.Alert([
        dbc.Row([
            dbc.Col([
                html.H5([html.I(className=f"bi {icon} me-2"), f"Verdict : {label}"],
                        className="mb-1"),
                html.Small(
                    f"{vote}/3 modeles predisent une resiliation." if vote >= 2
                    else f"{3-vote}/3 modeles predisent un client actif.",
                    style={"opacity": ".85"}
                ),
            ], md=7),
            dbc.Col([
                html.Div(f"Probabilite moyenne : {p_mean*100:.1f}%",
                         style={"fontWeight": "600", "marginBottom": "4px", "fontSize": ".9rem"}),
                html.Div(style={"background": "#e9ecef", "borderRadius": "6px", "height": "10px"},
                    children=html.Div(style={
                        "width": f"{p_mean*100:.1f}%", "height": "100%",
                        "background": "#dc3545" if vote >= 2 else "#198754",
                        "borderRadius": "6px",
                    })
                ),
            ], md=5, className="d-flex flex-column justify-content-center"),
        ], align="center"),
    ], color=color, className="fw-semibold")

    conf_msg = dbc.Alert([
        html.I(className=f"bi {'bi-exclamation-circle' if p_std > 0.15 else 'bi-check2-circle'} me-2"),
        (f"Les modeles sont en desaccord (ecart : {p_std*100:.1f}%). "
         "Client en zone d'incertitude — verification manuelle recommandee.")
        if p_std > 0.15 else
        f"Les trois modeles convergent (ecart : {p_std*100:.1f}%). Prediction fiable.",
    ], color="warning" if p_std > 0.15 else "light", className="small py-2 border")

    mlp_note = dbc.Alert([
        html.Div([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Pourquoi le MLP est-il plus variable ?"),
        ], className="mb-2"),
        html.Ul([
            html.Li([
                html.Strong("Architecture non lineaire : "),
                "le MLP (3 couches, 256-128-64 neurones) capte des interactions complexes "
                "entre attributs. Une variation sur NB_SERVICES ou VOL_APPELS peut traverser "
                "plusieurs couches et produire un saut de probabilite important.",
            ], className="mb-1"),
            html.Li([
                html.Strong("Entrainement sur donnees SMOTE : "),
                "le modele a ete entraine sur un jeu equilibre artificiellement (50 % resilie / "
                "50 % actif). Il a appris des frontieres de decision adaptees a cette distribution, "
                "ce qui le rend plus affirmatif que RF et XGBoost sur des cas particuliers.",
            ], className="mb-1"),
            html.Li([
                html.Strong("Calibration post-hoc : "),
                "une regression isotonique a ete apprise sur les donnees reelles (18 % de "
                "resiliation) pour corriger le biais de SMOTE. Malgre cela, le MLP reste "
                "sensible a certaines combinaisons de valeurs peu representees.",
            ], className="mb-1"),
            html.Li([
                html.Strong("Recommandation : "),
                "privilegier le consensus RF + XGBoost pour les decisions operationnelles. "
                "Le MLP apporte une information complementaire utile quand il converge avec les deux autres.",
            ]),
        ], className="mb-0 small"),
    ], color="light", className="small mt-2 border")

    legende = html.Div([
        html.Small([
            dbc.Badge("< 30%",  color="success", className="me-1"), "Faible  ",
            dbc.Badge("30-55%", color="warning", className="me-1"), "Modere  ",
            dbc.Badge("55-75%", color="danger",  className="me-1"), "Eleve  ",
            dbc.Badge("> 75%",  color="danger",  className="me-1"), "Critique",
        ], className="text-muted"),
    ], className="text-center mt-1 mb-2")

    return (
        [],
        fig_rf, fig_xgb, fig_mlp,
        gauge_style, gauge_style, gauge_style,
        html.Div([legende, conf_msg, consensus, mlp_note]),
    )


if __name__ == "__main__":
    print("\n  Dashboard disponible sur http://127.0.0.1:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
