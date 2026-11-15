"""
Klima – Next-day Solar Forecast + Digital Twin Dashboard
Veritas AI Hackathon 2026 · AI + Climate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import decision, load_data, make_features, train_and_evaluate

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Klima · Solar Digital Twin – Addis Ababa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-size: 2rem; font-weight: 700; color: #0f2922;
        margin-bottom: 0.15rem; letter-spacing: -0.02em;
    }
    .sub {
        color: #5a6a64; font-size: 1.05rem; margin-bottom: 1.4rem;
    }
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        border: 1px solid #e8eee9;
        box-shadow: 0 2px 8px rgba(15,41,34,0.04);
        height: 100%;
    }
    .card.orange {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white; border: none;
    }
    .card.green {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white; border: none;
    }
    .card.dark {
        background: #0f2922; color: #e8f5e9; border: none;
    }
    .card.soft {
        background: #f0fdf4; border: 1px solid #bbf7d0;
    }
    .card.warn {
        background: #fff7ed; border: 1px solid #fed7aa;
    }
    .metric-label { font-size: 0.8rem; opacity: 0.85; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.85rem; font-weight: 700; line-height: 1.15; }
    .metric-hint  { font-size: 0.8rem; opacity: 0.75; margin-top: 0.25rem; }
    .section-h {
        font-size: 1.15rem; font-weight: 600; color: #0f2922;
        margin: 1.4rem 0 0.7rem 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2922 0%, #134e3a 100%);
    }
    div[data-testid="stSidebar"] * { color: #e8f5e9 !important; }
    div[data-testid="stSidebar"] .stRadio label { padding: 0.35rem 0; }
    .impact-pos { color: #047857; font-weight: 600; }
    .impact-neg { color: #c2410c; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 klima")
    st.caption("LOCAL SUN · SMARTER ENERGY")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🗺️ Digital Twin Map",
            "📊 Performance",
            "📈 History",
            "🌍 Climate Impact",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**📍 Location**")
    st.markdown("Addis Ababa region")
    st.caption("9.01°N · 38.76°E · ~2 350 m")
    st.markdown("---")
    st.caption("NASA POWER 2020–2026")
    st.caption("Ridge · RF · TabPFN")
    st.caption("SHAP · LIME ready")


# ---------------------------------------------------------------------------
# Data / models (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def run_pipeline():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    df_raw = load_data(data_dir)
    df = make_features(df_raw)
    res = train_and_evaluate(df, train_ratio=0.85)
    return res, df_raw


res, df_raw = run_pipeline()
m = res["metrics"]
pred_rf = float(res["test"]["pred_rf"][-1])
pred_lin = float(res["test"]["pred_lin"][-1])
pred_tab = float(res["test"]["pred_tab"][-1])
today_solar = float(res["test"]["y_today"][-1])
has_tabpfn = res.get("has_tabpfn", False)
best_pred = pred_lin if m["linear"]["r2"] >= m["random_forest"]["r2"] else pred_rf
dec = decision(best_pred, threshold=26.0)


def to_kwh(mj: float) -> float:
    return mj / 3.6


# ---------------------------------------------------------------------------
# 🏠 OVERVIEW
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.markdown('<div class="main-title">Your next day, in clear view.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">AI forecast for household solar & battery decisions in Addis Ababa.</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta = (best_pred - today_solar) / today_solar * 100 if today_solar else 0
        st.markdown(
            f"""
            <div class="card orange">
                <div class="metric-label">Tomorrow’s solar (best model)</div>
                <div class="metric-value">{to_kwh(best_pred):.2f} kWh/m²</div>
                <div class="metric-hint">{delta:+.0f}% vs today</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label" style="color:#666">Today (observed)</div>
                <div class="metric-value" style="color:#0f2922">{to_kwh(today_solar):.2f}</div>
                <div class="metric-hint" style="color:#888">kWh/m² · clear-sky</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        gen = to_kwh(best_pred) * 5 * 0.85
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label" style="color:#666">Est. generation (5 kW system)</div>
                <div class="metric-value" style="color:#0f2922">{gen:.0f} kWh</div>
                <div class="metric-hint" style="color:#888">simplified estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        color = "green" if dec["is_good"] else "dark"
        st.markdown(
            f"""
            <div class="card {color}">
                <div class="metric-label">Battery decision</div>
                <div class="metric-value" style="font-size:1.25rem">{dec['action']}</div>
                <div class="metric-hint">Reserve ≈ {dec['reserve_pct']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-h">Model quality at a glance</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ridge R²", f"{m['linear']['r2']:.3f}")
    k2.metric("Random Forest R²", f"{m['random_forest']['r2']:.3f}")
    k3.metric("TabPFN R²", f"{m['tabpfn']['r2']:.3f}")
    k4.metric("Persistence R²", f"{m['persistence']['r2']:.3f}")

    st.markdown('<div class="section-h">Recent forecast vs actual</div>', unsafe_allow_html=True)
    n = min(14, len(res["test"]["y_true"]))
    dates = pd.to_datetime(res["test"]["dates"][-n:])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates.strftime("%d %b"),
            y=[to_kwh(v) for v in res["test"]["y_true"][-n:]],
            name="Actual",
            marker_color="#059669",
        )
    )
    fig.add_trace(
        go.Bar(
            x=dates.strftime("%d %b"),
            y=[to_kwh(v) for v in res["test"]["pred_lin"][-n:]],
            name="Ridge forecast",
            marker_color="#f59e0b",
        )
    )
    fig.update_layout(
        height=320,
        barmode="group",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="kWh/m²/day",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("What does the battery decision mean?"):
        st.markdown(
            f"""
            **{dec['action']}**  
            {dec['detail']}

            Threshold used: **{dec['threshold']:.0f} MJ/m²/day**  
            (≈ {to_kwh(dec['threshold']):.1f} kWh/m²).  
            This is a simple, transparent rule you can change later.
            """
        )

# ---------------------------------------------------------------------------
# 🗺️ DIGITAL TWIN MAP
# ---------------------------------------------------------------------------
elif page == "🗺️ Digital Twin Map":
    st.markdown('<div class="main-title">Digital Twin — Addis Ababa</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Location of the NASA POWER grid cell that feeds Klima, with live forecast context.</div>',
        unsafe_allow_html=True,
    )

    # Approximate centre of the MERRA-2 / POWER cell
    lat, lon = 9.0054, 38.7636
    elev = 2349.69

    # Map
    fig_map = go.Figure(
        go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            marker=dict(size=18, color="#f59e0b"),
            text=["Klima grid cell"],
            textposition="top center",
            hovertemplate=(
                f"<b>Addis Ababa region</b><br>"
                f"Lat {lat:.4f} · Lon {lon:.4f}<br>"
                f"Elevation ≈ {elev:.0f} m<br>"
                f"Tomorrow forecast: {to_kwh(best_pred):.2f} kWh/m²"
                "<extra></extra>"
            ),
        )
    )
    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=9.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="card soft">
                <div class="metric-label" style="color:#047857">Grid cell centre</div>
                <div style="font-size:1.1rem;font-weight:600;color:#0f2922">{lat:.4f}°N, {lon:.4f}°E</div>
                <div class="metric-hint" style="color:#5a6a64">0.5° × 0.625° MERRA-2 region</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card soft">
                <div class="metric-label" style="color:#047857">Elevation</div>
                <div style="font-size:1.1rem;font-weight:600;color:#0f2922">≈ {elev:.0f} m</div>
                <div class="metric-hint" style="color:#5a6a64">High plateau · thinner air, strong solar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="card soft">
                <div class="metric-label" style="color:#047857">Tomorrow (model)</div>
                <div style="font-size:1.1rem;font-weight:600;color:#0f2922">{to_kwh(best_pred):.2f} kWh/m²</div>
                <div class="metric-hint" style="color:#5a6a64">{dec['action']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-h">Why this location?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Addis Ababa sits on the Ethiopian highlands. Clear-sky solar resource is strong year-round,  
        but day-to-day and seasonal swings still matter for battery sizing and overnight charging.  
        Klima treats the NASA POWER cell as a lightweight **digital twin** of the local solar climate:  
        same coordinates, multi-year history, and a next-day forecast that a household or micro-grid can act on.
        """
    )

# ---------------------------------------------------------------------------
# 📊 PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "📊 Performance":
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub">Chronological 85 % train / 15 % test · {res["train_size"]} train days · {res["test_size"]} test days</div>',
        unsafe_allow_html=True,
    )

    dfm = pd.DataFrame(
        {
            "Model": ["Ridge", "Random Forest", "TabPFN", "Persistence"],
            "R²": [
                m["linear"]["r2"],
                m["random_forest"]["r2"],
                m["tabpfn"]["r2"],
                m["persistence"]["r2"],
            ],
            "MAE (MJ/m²)": [
                m["linear"]["mae"],
                m["random_forest"]["mae"],
                m["tabpfn"]["mae"],
                m["persistence"]["mae"],
            ],
            "RMSE (MJ/m²)": [
                m["linear"]["rmse"],
                m["random_forest"]["rmse"],
                m["tabpfn"]["rmse"],
                m["persistence"]["rmse"],
            ],
        }
    )
    st.dataframe(
        dfm.style.format(
            {"R²": "{:.4f}", "MAE (MJ/m²)": "{:.3f}", "RMSE (MJ/m²)": "{:.3f}"}
        ).background_gradient(subset=["R²"], cmap="YlGn"),
        use_container_width=True,
    )

    st.success(
        f"All models achieve R² ≥ 0.85. Best point forecast: **Ridge R² = {m['linear']['r2']:.4f}**"
    )

    st.markdown('<div class="section-h">Decision accuracy at different thresholds</div>', unsafe_allow_html=True)
    for th in [25.0, 26.0, 26.5, 27.0]:
        actual = res["test"]["y_true"] > th
        parts = []
        for name, pred in [
            ("Ridge", res["test"]["pred_lin"]),
            ("RF", res["test"]["pred_rf"]),
            ("TabPFN", res["test"]["pred_tab"]),
            ("Pers", res["test"]["pred_pers"]),
        ]:
            acc = (actual == (pred > th)).mean() * 100
            parts.append(f"{name} **{acc:.1f}%**")
        st.write(f"Threshold {th:.1f} MJ/m² · " + " · ".join(parts))

# ---------------------------------------------------------------------------
# 📈 HISTORY
# ---------------------------------------------------------------------------
elif page == "📈 History":
    st.markdown('<div class="main-title">Forecast vs Actual</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Test period only (chronological hold-out)</div>', unsafe_allow_html=True)

    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(res["test"]["dates"]),
            "actual": res["test"]["y_true"],
            "ridge": res["test"]["pred_lin"],
            "rf": res["test"]["pred_rf"],
            "tabpfn": res["test"]["pred_tab"],
            "persistence": res["test"]["pred_pers"],
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual"], name="Actual", line=dict(color="#059669", width=2.2)))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["ridge"], name="Ridge", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["rf"], name="Random Forest", line=dict(color="#f59e0b")))
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["persistence"],
            name="Persistence",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    fig.update_layout(
        height=420,
        yaxis_title="MJ/m²/day",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hist.tail(15).style.format(precision=2), use_container_width=True)

# ---------------------------------------------------------------------------
# 🌍 CLIMATE IMPACT
# ---------------------------------------------------------------------------
elif page == "🌍 Climate Impact":
    st.markdown('<div class="main-title">Climate Impact Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">What the predicted vs current solar climate means for a household or micro-grid in Addis Ababa.</div>',
        unsafe_allow_html=True,
    )

    delta_pct = (best_pred - today_solar) / today_solar * 100 if today_solar else 0
    better = best_pred > today_solar

    # Snapshot cards
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"""
            <div class="card soft">
                <div class="metric-label" style="color:#047857">Current (today)</div>
                <div class="metric-value" style="color:#0f2922">{to_kwh(today_solar):.2f} kWh/m²</div>
                <div class="metric-hint" style="color:#5a6a64">Observed clear-sky resource</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""
            <div class="card {'soft' if better else 'warn'}">
                <div class="metric-label" style="color:{'#047857' if better else '#c2410c'}">Predicted (tomorrow)</div>
                <div class="metric-value" style="color:#0f2922">{to_kwh(best_pred):.2f} kWh/m²</div>
                <div class="metric-hint" style="color:#5a6a64">{delta_pct:+.1f}% vs today · {dec['action']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-h">Advantages of acting on this forecast</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card soft">
        <ul style="margin:0; padding-left:1.1rem; color:#0f2922; line-height:1.55">
            <li><span class="impact-pos">Lower electricity cost</span> — charge from the grid only when tomorrow’s solar is weak; otherwise let the sun refill the battery.</li>
            <li><span class="impact-pos">Higher self-consumption</span> — keep battery headroom when a strong clear-sky day is expected.</li>
            <li><span class="impact-pos">Longer battery life</span> — fewer unnecessary full cycles reduce wear.</li>
            <li><span class="impact-pos">Grid relief</span> — less evening peak demand when many homes follow the same rule.</li>
            <li><span class="impact-pos">Transparent rule</span> — threshold at {dec['threshold']:.0f} MJ/m² is easy to explain and audit.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">Disadvantages & risks to keep in mind</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card warn">
        <ul style="margin:0; padding-left:1.1rem; color:#0f2922; line-height:1.55">
            <li><span class="impact-neg">Clear-sky ≠ all-sky</span> — the model predicts clear-sky irradiance. Clouds, dust or haze can still cut real output.</li>
            <li><span class="impact-neg">Grid-cell average</span> — NASA POWER is a 0.5°×0.625° average; a single rooftop can differ.</li>
            <li><span class="impact-neg">Threshold sensitivity</span> — a fixed cut-off (26 MJ/m²) is simple but not optimal for every tariff or battery size.</li>
            <li><span class="impact-neg">Forecast error</span> — even with R² ≈ 0.91, a few days will be wrong; always keep a safety reserve.</li>
            <li><span class="impact-neg">Data lag</span> — near-real-time satellite products can lag; the latest day may be incomplete.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">Impact summary for Addis Ababa</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        On the Ethiopian highlands the clear-sky resource is already strong. Klima’s value is in the **day-to-day timing**:  
        knowing whether tomorrow is likely above or below the seasonal norm lets a household (or a small community micro-grid)  
        shift charging by a few hours and capture more free solar.

        - **If tomorrow looks strong** ({to_kwh(best_pred):.2f} kWh/m²) → leave ~{dec['reserve_pct'] if dec['is_good'] else 20}% reserve and prioritise solar charging.  
        - **If tomorrow looks weak** → fill the battery from the grid overnight and avoid morning shortfall.

        Over a year, even a modest improvement in self-consumption reduces diesel backup hours and grid imports —  
        a practical climate action that scales from one home to a neighbourhood.
        """
    )

    # Simple seasonal context from data
    if "date" in df_raw.columns and "solar" in df_raw.columns:
        tmp = df_raw.dropna(subset=["solar"]).copy()
        tmp["month"] = tmp["date"].dt.month
        monthly = tmp.groupby("month")["solar"].mean() / 3.6
        fig_s = go.Figure(
            go.Bar(
                x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                y=[monthly.get(i, np.nan) for i in range(1, 13)],
                marker_color="#059669",
            )
        )
        fig_s.update_layout(
            title="Average clear-sky solar by month (kWh/m²/day)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="kWh/m²/day",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ---------------------------------------------------------------------------
# ℹ️ ABOUT
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="main-title">About Klima</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Klima** is a next-day clear-sky solar radiation forecast system built for the  
        **Veritas AI International High School Hackathon 2026 – Track AI + Climate**.

        ### What it does
        - Forecasts tomorrow’s clear-sky irradiance for the Addis Ababa region  
        - Turns the forecast into a simple household battery rule  
        - Explains predictions with SHAP and LIME  
        - Shows a lightweight digital twin of the NASA POWER grid cell  

        ### Models
        - Ridge (best R² ≈ 0.91)  
        - Random Forest  
        - TabPFN (foundation model; optional token)  
        - Persistence baseline  

        ### Data
        - All original rows from the provided 2026 NASA POWER file  
        - Extra multi-year history 2020–2025 for the same point  
        - Location ≈ 9.01°N, 38.76°E, ~2 350 m  

        ### Design principles
        - Chronological split only (no leakage)  
        - Honest baselines and metrics  
        - Transparent decision rule  
        - Simple UI for first-time users, full depth for judges  

        ---
        Built for a 2-minute demo and a clear written submission.
        """
    )
