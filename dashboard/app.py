"""
SAP Intelligent Demand Forecasting - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys
import plotly.graph_objects as go
import plotly.express as px
import shap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
SRC  = os.path.join(ROOT, "src")

# Ensure data & model exist (runs data gen + training on first load)
sys.path.insert(0, ROOT)
import startup; startup.ensure_artifacts()

st.set_page_config(
    page_title="SAP Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .kpi-card {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 16px 20px;
    border-left: 4px solid #0a6ed1;
  }
  .kpi-label { font-size: 11px; color: #9ba3ad; font-weight: 500; text-transform: uppercase; letter-spacing:.5px; }
  .kpi-value { font-size: 26px; font-weight: 700; color: #e8eaed; margin-top: 4px; }
  .kpi-sub   { font-size: 11px; color: #4caf7d; margin-top: 2px; }
  .section-header {
    font-size: 14px; font-weight: 600; color: #9ba3ad;
    border-bottom: 1px solid #2e3440; padding-bottom: 6px; margin: 24px 0 12px;
    text-transform: uppercase; letter-spacing: .6px;
  }
  .sap-badge {
    display: inline-block; background: #1a2a3a; color: #4d9de0;
    border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ── Shared axis style — gray ticks, transparent bg ──────────────────────
AXIS = dict(
    color="#9ba3ad",           # gray tick labels
    tickfont=dict(size=11, color="#9ba3ad"),
    linecolor="#3a3f4b",
    gridcolor="#2e3440",
    zerolinecolor="#3a3f4b",
)
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9ba3ad"),
)

# ── Load data & model ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df      = pd.read_csv(f"{DATA}/monthly_demand_features.csv", parse_dates=["YEAR_MONTH"])
    mara    = pd.read_csv(f"{DATA}/MARA.csv")
    metrics = pd.read_csv(f"{DATA}/model_metrics.csv")
    return df, mara, metrics

@st.cache_resource
def load_model():
    with open(f"{SRC}/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{SRC}/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, explainer

df, mara, metrics = load_data()
model, explainer   = load_model()

FEATURES = [
    "MONTH","QUARTER","YEAR","IS_Q4","IS_SUMMER","MATKL_ENC",
    "LAG_1","LAG_2","LAG_3","LAG_6","LAG_12",
    "ROLL_3_MEAN","ROLL_6_MEAN","ROLL_3_STD"
]

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sap-badge">SAP SD Module</div>', unsafe_allow_html=True)
    st.title("Demand Forecasting")
    st.caption("Powered by XGBoost + SHAP")
    st.divider()
    mat_list     = sorted(df["MATNR"].unique())
    selected_mat = st.selectbox("Material (MATNR)", mat_list)
    mat_info     = mara[mara["MATNR"] == selected_mat].iloc[0]
    st.info(f"**Group:** {mat_info['MATKL']}  \n**Type:** {mat_info['MTART']}  \n**Unit:** {mat_info['MEINS']}")
    st.divider()
    forecast_months = st.slider("Forecast horizon (months)", 1, 12, 6)
    st.caption("SAP tables used: VBAK · VBAP · MARA · KNA1 · T001W")

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(f"## 📦 SAP Intelligent Demand Forecasting")
st.markdown(f"Material: **{selected_mat}** — {mat_info['MAKTX']}")

# ── KPI Row ──────────────────────────────────────────────────────────────
mat_df       = df[df["MATNR"] == selected_mat].sort_values("YEAR_MONTH")
total_demand = int(mat_df["DEMAND"].sum())
avg_monthly  = int(mat_df["DEMAND"].mean())
last_demand  = int(mat_df["DEMAND"].iloc[-1]) if len(mat_df) else 0
mape_val     = float(metrics["MAPE"].iloc[0])

c1, c2, c3, c4 = st.columns(4)
for col, label, value, sub in [
    (c1, "Total demand (all time)", f"{total_demand:,}", "units sold"),
    (c2, "Avg monthly demand",      f"{avg_monthly:,}",  "units / month"),
    (c3, "Last month actual",       f"{last_demand:,}",  "most recent period"),
    (c4, "Model MAPE",              f"{mape_val:.1%}",   "XGBoost forecast error"),
]:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Forecast chart ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Demand forecast vs actual</div>', unsafe_allow_html=True)

if len(mat_df) < 13:
    st.warning("Not enough history for this material — select one with more data.")
else:
    last_row    = mat_df.iloc[-1]
    future_rows = []
    for i in range(1, forecast_months + 1):
        fut_date = mat_df["YEAR_MONTH"].iloc[-1] + pd.DateOffset(months=i)
        row = {
            "YEAR_MONTH":  fut_date,
            "MONTH":       fut_date.month,
            "QUARTER":     fut_date.quarter,
            "YEAR":        fut_date.year,
            "IS_Q4":       int(fut_date.quarter == 4),
            "IS_SUMMER":   int(fut_date.month in [6,7,8]),
            "MATKL_ENC":   last_row["MATKL_ENC"],
            "LAG_1":       last_row["DEMAND"] if i == 1 else future_rows[-1]["DEMAND_PRED"],
            "LAG_2":       mat_df["DEMAND"].iloc[-1] if i <= 1 else (last_row["DEMAND"] if i == 2 else future_rows[-2]["DEMAND_PRED"]),
            "LAG_3":       mat_df["DEMAND"].iloc[-1],
            "LAG_6":       mat_df["DEMAND"].iloc[-6]  if len(mat_df) >= 6  else last_row["DEMAND"],
            "LAG_12":      mat_df["DEMAND"].iloc[-12] if len(mat_df) >= 12 else last_row["DEMAND"],
            "ROLL_3_MEAN": mat_df["DEMAND"].iloc[-3:].mean(),
            "ROLL_6_MEAN": mat_df["DEMAND"].iloc[-6:].mean(),
            "ROLL_3_STD":  mat_df["DEMAND"].iloc[-3:].std(),
        }
        pred = float(np.clip(model.predict(pd.DataFrame([row])[FEATURES]), 0, None).item())
        row["DEMAND_PRED"] = pred
        future_rows.append(row)

    fut_df          = pd.DataFrame(future_rows)
    in_sample_pred  = np.clip(model.predict(mat_df[FEATURES].dropna()), 0, None)
    in_sample_x     = mat_df["YEAR_MONTH"].iloc[len(mat_df)-len(in_sample_pred):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mat_df["YEAR_MONTH"], y=mat_df["DEMAND"],
        name="Actual", line=dict(color="#4d9de0", width=2),
        mode="lines",
        hovertemplate="%{x|%b %Y}  Actual: <b>%{y:,.0f}</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=in_sample_x, y=in_sample_pred,
        name="Model fit", line=dict(color="#6b7280", width=1.2, dash="dot"),
        hovertemplate="%{x|%b %Y}  Fit: <b>%{y:,.0f}</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=fut_df["YEAR_MONTH"], y=fut_df["DEMAND_PRED"],
        name="Forecast", line=dict(color="#4caf7d", width=2),
        mode="lines+markers", marker=dict(size=6, symbol="diamond", color="#4caf7d"),
        hovertemplate="%{x|%b %Y}  Forecast: <b>%{y:,.0f}</b><extra></extra>"
    ))
    fig.add_vline(
        x=mat_df["YEAR_MONTH"].iloc[-1].timestamp() * 1000,
        line_width=1, line_dash="dash", line_color="#4b5563",
        annotation_text="Forecast →",
        annotation_font_size=11, annotation_font_color="#9ba3ad",
        annotation_position="top right"
    )
    fig.update_layout(
        **LAYOUT,
        height=380,
        margin=dict(l=70, r=20, t=50, b=60),
        legend=dict(
            orientation="h", y=1.12, x=0,
            font=dict(size=12, color="#9ba3ad"),
            bgcolor="rgba(0,0,0,0)"
        ),
        hovermode="x unified",
        xaxis=dict(**AXIS,
            showgrid=False, showline=True,
            tickformat="%b %Y", tickangle=-30,
            title=dict(text="Month", font=dict(size=12, color="#9ba3ad")),
        ),
        yaxis=dict(**AXIS,
            showgrid=True, zeroline=False,
            tickformat=",",
            title=dict(text="Quantity (units)", font=dict(size=12, color="#9ba3ad")),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View forecast table (SAP-style output)"):
        tbl = fut_df[["YEAR_MONTH","DEMAND_PRED"]].copy()
        tbl.columns = ["Period", "Forecast Qty"]
        tbl["Period"]       = tbl["Period"].dt.strftime("%Y-%m")
        tbl["Forecast Qty"] = tbl["Forecast Qty"].round(0).astype(int)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── SHAP chart ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Feature importance (SHAP)</div>', unsafe_allow_html=True)
st.caption("Which features drove this forecast? SHAP values show each feature's average impact on model output.")

X_sample  = df[FEATURES].dropna().sample(min(200, len(df)), random_state=42)
shap_vals = explainer.shap_values(X_sample)
name_map  = {
    "LAG_1": "Last month demand",    "LAG_2": "2 months ago",
    "LAG_3": "3 months ago",         "LAG_6": "6 months ago",
    "LAG_12": "12 months ago",       "ROLL_3_MEAN": "3-month avg",
    "ROLL_6_MEAN": "6-month avg",    "ROLL_3_STD": "3-month volatility",
    "MONTH": "Month of year",        "QUARTER": "Quarter",
    "YEAR": "Year trend",            "IS_Q4": "Q4 flag",
    "IS_SUMMER": "Summer flag",      "MATKL_ENC": "Material group"
}
mean_shap = pd.DataFrame({
    "Feature":    FEATURES,
    "Label":      [name_map.get(f, f) for f in FEATURES],
    "Mean |SHAP|": np.abs(shap_vals).mean(axis=0)
})
mean_shap = mean_shap[mean_shap["Feature"] != "YEAR"]
mean_shap = mean_shap.sort_values("Mean |SHAP|", ascending=True)

n      = len(mean_shap)
colors = ["#3a4a5a"] * n
for i in range(max(0, n-3), n):
    colors[i] = "#4d9de0"

bar_labels = [f"{v:.1f}" for v in mean_shap["Mean |SHAP|"]]

fig2 = go.Figure(go.Bar(
    x=mean_shap["Mean |SHAP|"],
    y=mean_shap["Label"],
    orientation="h",
    marker_color=colors,
    marker_line_width=0,
    text=bar_labels,
    textposition="outside",
    textfont=dict(size=11, color="#9ba3ad"),
    hovertemplate="<b>%{y}</b>  Impact: %{x:.1f}<extra></extra>"
))
fig2.update_layout(
    **LAYOUT,
    height=400,
    margin=dict(l=160, r=70, t=20, b=50),
    xaxis=dict(**AXIS,
        showgrid=False, zeroline=False, showline=True,
        title=dict(text="Mean absolute SHAP value", font=dict(size=12, color="#9ba3ad")),
    ),
    yaxis=dict(**AXIS,
        showgrid=False, showline=False,
    ),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption("'Year trend' excluded — it captures dataset time span, not a real business driver.")

# ── Heatmap ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Demand by material group over time</div>', unsafe_allow_html=True)

heat             = df.groupby(["MATKL", df["YEAR_MONTH"].dt.year])["DEMAND"].sum().reset_index()
heat.columns     = ["Material Group", "Year", "Demand"]
heat["Year"]     = heat["Year"].astype(int)
pivot            = heat.pivot(index="Material Group", columns="Year", values="Demand").fillna(0).astype(int)

fig3 = px.imshow(
    pivot,
    color_continuous_scale=[[0,"#1a2535"],[0.5,"#1e5fa3"],[1.0,"#4d9de0"]],
    aspect="auto",
    text_auto=False,
)
# Overlay text manually with correct color contrast
for i, row_name in enumerate(pivot.index):
    for j, col_name in enumerate(pivot.columns):
        val = pivot.loc[row_name, col_name]
        fig3.add_annotation(
            x=col_name, y=row_name,
            text=f"{val:,}",
            showarrow=False,
            font=dict(size=11, color="#e8eaed"),
            xref="x", yref="y"
        )
fig3.update_layout(
    **LAYOUT,
    height=260,
    margin=dict(l=70, r=20, t=10, b=50),
    xaxis=dict(**AXIS,
        tickmode="array",
        tickvals=list(pivot.columns),
        ticktext=[str(y) for y in pivot.columns],
        title=dict(text="Year", font=dict(size=12, color="#9ba3ad")),
        showgrid=False,
    ),
    yaxis=dict(**AXIS,
        showgrid=False, showline=False,
        title="",
    ),
    coloraxis_showscale=False,
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption("Data: Synthetic SAP SD/MM tables (VBAK · VBAP · MARA · KNA1 · T001W) · Model: XGBoost · Explainability: SHAP · Built with Python + Streamlit")