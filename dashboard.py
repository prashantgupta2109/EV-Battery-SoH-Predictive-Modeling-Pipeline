# =============================================================================
#  dashboard.py
#  EV Battery State-of-Health (SoH) Interactive Dashboard
#  Run with: streamlit run dashboard.py
#
#  Tab 1 — 🔮 Real-Time Health Predictor
#  Tab 2 — 📊 Battery Degradation Analytics
# =============================================================================

import os
import glob
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Battery Health Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Base & fonts ---------- */
    /*
     * Inter is loaded from Google Fonts when online.
     * Falls back to system UI fonts when there is no internet connection,
     * so the dashboard stays fully functional on offline / air-gapped machines.
     */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                     Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue',
                     Arial, sans-serif;
    }

    /* ---------- App background ---------- */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #1c2128 100%);
        border-right: 1px solid #30363d;
    }

    /* ---------- Tab bar ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #8b949e;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #238636, #1a7f37) !important;
        color: #ffffff !important;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #58a6ff !important;
    }

    /* ---------- Primary button ---------- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636, #1a7f37);
        border: none;
        border-radius: 10px;
        padding: 14px 36px;
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.5px;
        transition: all 0.25s ease;
        width: 100%;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(35, 134, 54, 0.45);
    }

    /* ---------- Slider ---------- */
    .stSlider [data-testid="stSlider"] > div > div > div {
        background: linear-gradient(90deg, #238636, #58a6ff) !important;
    }

    /* ---------- Info / warning / error boxes ---------- */
    .stAlert {
        border-radius: 10px;
    }

    /* ---------- Number inputs ---------- */
    .stNumberInput > div {
        border-radius: 8px;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid #238636;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* ---------- SoH gauge badge ---------- */
    .soh-badge-ok {
        background: linear-gradient(135deg, #238636, #1a7f37);
        color: white;
        border-radius: 12px;
        padding: 20px 30px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(35, 134, 54, 0.4);
        animation: pulse-green 2s ease-in-out infinite;
    }
    .soh-badge-warn {
        background: linear-gradient(135deg, #b91c1c, #dc2626);
        color: white;
        border-radius: 12px;
        padding: 20px 30px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(185, 28, 28, 0.5);
        animation: pulse-red 1s ease-in-out infinite;
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(35, 134, 54, 0.3); }
        50%       { box-shadow: 0 0 40px rgba(35, 134, 54, 0.7); }
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(185, 28, 28, 0.4); }
        50%       { box-shadow: 0 0 45px rgba(185, 28, 28, 0.9); }
    }

    /* ---------- Dividers ---------- */
    hr { border-color: #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Resolve ALL paths relative to THIS script file, not the shell CWD.
# This guarantees the dashboard finds its data and model artifacts whether
# it is launched from the project directory, a parent folder, or any other
# working directory on any machine.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR  = os.path.join(BASE_DIR, "data_battery_cycle")

MODEL_PATH   = os.path.join(MODEL_DIR, "battery_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")


# ---------------------------------------------------------------------------
# Plotly dark template shared across all charts
# ---------------------------------------------------------------------------
# PX_ARGS  → passed directly to px.line / px.scatter / px.area / px.histogram / px.box
#             (only accepts 'template', NOT paper_bgcolor / plot_bgcolor)
PX_ARGS = dict(template="plotly_dark")

# LAYOUT_STYLE → spread inside update_layout() to apply transparent backgrounds
LAYOUT_STYLE = dict(
    paper_bgcolor="rgba(22,27,34,0)",
    plot_bgcolor ="rgba(22,27,34,0)",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    """Load the trained model, scaler, feature list, and metrics from model/."""
    try:
        model         = joblib.load(MODEL_PATH)
        scaler        = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_PATH)
        metrics       = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else {}
        return model, scaler, feature_names, metrics
    except FileNotFoundError:
        return None, None, None, {}


@st.cache_data(show_spinner="Loading battery dataset...")
def load_battery_data() -> pd.DataFrame:
    """Load the first CSV found in data_battery_cycle/."""
    pattern   = os.path.join(DATA_DIR, "*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        return pd.DataFrame()
    df = pd.read_csv(csv_files[0])
    df.columns = [c.lower().strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Load everything once at startup
# ---------------------------------------------------------------------------
model, scaler, feature_names, metrics = load_model_artifacts()
df_raw = load_battery_data()


# ---------------------------------------------------------------------------
# ════════════════════  SIDEBAR  ════════════════════
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 10px 0 20px;'>"
        "<span style='font-size:3rem;'>🔋</span>"
        "<h2 style='color:#58a6ff; margin:0; font-size:1.3rem;'>Battery Health Engine</h2>"
        "<p style='color:#8b949e; font-size:0.8rem; margin:4px 0 0;'>LightGBM · NASA CALCE Dataset</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Model status
    if model is not None:
        st.success("✅  Model artifacts loaded")
        if metrics:
            st.metric("R² Accuracy",  f"{metrics['r2'] * 100:.2f}%")
            st.metric("RMSE",         f"{metrics['rmse']:.5f}")
        st.markdown(f"**Features:** `{', '.join(feature_names)}`")
    else:
        st.error("🚨 Model not found — run `python train_pipeline.py` first.")

    st.divider()

    # Dataset info
    if not df_raw.empty:
        st.success(f"✅  Dataset loaded  ({len(df_raw):,} rows)")
        if "battery_id" in df_raw.columns:
            batteries = sorted(df_raw["battery_id"].unique())
            st.markdown(f"**Batteries:** {', '.join(batteries)}")
    else:
        st.warning("⚠️ No dataset found in `data_battery_cycle/`.")

    st.divider()
    st.caption("© 2026 Battery Analytics Dashboard")


# ---------------------------------------------------------------------------
# ════════════════════  HEADER  ════════════════════
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; font-size:2.6rem; font-weight:800; "
    "background: linear-gradient(90deg, #58a6ff, #3fb950, #f78166); "
    "-webkit-background-clip:text; -webkit-text-fill-color:transparent; "
    "margin-bottom:4px;'>⚡ EV Battery State-of-Health Dashboard</h1>"
    "<p style='text-align:center; color:#8b949e; font-size:1rem; margin-bottom:24px;'>"
    "Real-time SoH prediction & degradation analytics powered by LightGBM</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# ════════════════════  TABS  ════════════════════
# ---------------------------------------------------------------------------
tab_predictor, tab_analytics = st.tabs(
    ["🔮 Real-Time Health Predictor", "📊 Battery Degradation Analytics"]
)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_predictor:
    if model is None:
        st.error(
            "🚨 **Model artifacts not found.**  "
            "Please run `python train_pipeline.py` from your terminal first, "
            "then refresh this page."
        )
        st.stop()

    st.markdown(
        "<div class='section-header'>⚙️ Set Operational Parameters</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Adjust the battery's current operational telemetry using the sliders below. "
        "Click **Predict SoH** to receive a real-time health estimate from the trained model."
    )
    st.markdown("")

    # --- Input widgets: 3-column layout ---
    col_v, col_t, col_c = st.columns(3, gap="large")

    with col_v:
        st.markdown("#### 🔌 Average Voltage (V)")
        st.caption("Typical Li-Ion discharge range: 3.2 V – 4.2 V")
        voltage = st.slider(
            label="Average Voltage",
            min_value=2.5,
            max_value=4.5,
            value=3.55,
            step=0.005,
            format="%.3f V",
            label_visibility="collapsed",
            key="voltage_slider",
        )
        voltage_num = st.number_input(
            "Fine-tune voltage",
            min_value=2.5,
            max_value=4.5,
            value=voltage,
            step=0.001,
            format="%.3f",
            key="voltage_num",
        )
        voltage = voltage_num  # number input overrides slider

    with col_t:
        st.markdown("#### 🌡️ Core Temperature (°C)")
        st.caption("Safe operating range: 20 °C – 60 °C")
        temperature = st.slider(
            label="Core Temperature",
            min_value=15.0,
            max_value=75.0,
            value=32.5,
            step=0.5,
            format="%.1f °C",
            label_visibility="collapsed",
            key="temp_slider",
        )
        temperature_num = st.number_input(
            "Fine-tune temperature",
            min_value=15.0,
            max_value=75.0,
            value=temperature,
            step=0.1,
            format="%.1f",
            key="temp_num",
        )
        temperature = temperature_num

    with col_c:
        st.markdown("#### 🔄 Charge Cycle Count")
        st.caption("Higher cycles → greater wear on active material")
        cycle = st.slider(
            label="Cycle Count",
            min_value=1,
            max_value=300,
            value=50,
            step=1,
            format="%d cycles",
            label_visibility="collapsed",
            key="cycle_slider",
        )
        cycle_num = st.number_input(
            "Fine-tune cycle count",
            min_value=1,
            max_value=300,
            value=cycle,
            step=1,
            key="cycle_num",
        )
        cycle = int(cycle_num)

    st.markdown("")
    st.divider()

    # --- Predict button ---
    col_btn, col_result = st.columns([1, 2], gap="large")

    with col_btn:
        predict_clicked = st.button(
            "⚡ Predict SoH",
            type="primary",
            key="predict_btn",
        )

    with col_result:
        if predict_clicked:
            # Build a named DataFrame — same column order used during training.
            # Using a DataFrame (not a numpy array) suppresses the sklearn
            # UserWarning: "X does not have valid feature names" on all systems.
            input_df  = pd.DataFrame(
                [[voltage, temperature, cycle]],
                columns=feature_names,
            )
            scaled_df = pd.DataFrame(
                scaler.transform(input_df),
                columns=feature_names,
            )
            soh_raw   = float(model.predict(scaled_df)[0])

            # Clamp prediction to a sensible display range
            soh_display = np.clip(soh_raw, 0.0, 1.0)
            soh_pct     = soh_display * 100.0

            # --- SoH gauge ---
            st.markdown("<br>", unsafe_allow_html=True)

            if soh_pct >= 80.0:
                badge_class = "soh-badge-ok"
                status_icon = "✅"
                status_text = "NOMINAL — Battery is healthy and within safe operating range."
            elif soh_pct >= 60.0:
                badge_class = "soh-badge-warn"
                status_icon = "⚠️"
                status_text = "DEGRADED — Battery shows moderate wear. Schedule inspection soon."
            else:
                badge_class = "soh-badge-warn"
                status_icon = "🚨"
                status_text = "CRITICAL — Battery capacity severely degraded. Immediate replacement advised!"

            st.markdown(
                f"<div class='{badge_class}'>"
                f"State of Health: {soh_pct:.1f}%"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if soh_pct < 80.0:
                st.error(
                    f"{status_icon} **CRITICAL WARNING** — SoH is **{soh_pct:.1f}%**, "
                    f"which is below the 80% end-of-life threshold. {status_text}"
                )
            else:
                st.success(f"{status_icon} **{status_text}**")

    # --- Input summary cards ---
    if predict_clicked:
        st.divider()
        st.markdown("**Telemetry submitted to model:**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Average Voltage",  f"{voltage:.3f} V")
        m2.metric("Core Temperature", f"{temperature:.1f} °C")
        m3.metric("Cycle Count",      f"{cycle}")

    # --- Gauge chart (always visible) ---
    st.divider()
    st.markdown("<div class='section-header'>📈 Live SoH Gauge</div>", unsafe_allow_html=True)

    if predict_clicked:
        gauge_val = soh_pct
    else:
        gauge_val = 100.0  # default when no prediction yet

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": 80, "suffix": "%",
               "increasing": {"color": "#3fb950"},
               "decreasing": {"color": "#f85149"}},
        number={"suffix": "%", "font": {"size": 48, "color": "#58a6ff"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#8b949e", "dtick": 10},
            "bar": {"color": "#58a6ff", "thickness": 0.25},
            "bgcolor": "#161b22",
            "borderwidth": 2,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  60], "color": "#3d0000"},
                {"range": [60, 80], "color": "#3d2200"},
                {"range": [80, 100],"color": "#0d2b10"},
            ],
            "threshold": {
                "line": {"color": "#f85149", "width": 4},
                "thickness": 0.85,
                "value": 80,
            },
        },
        title={"text": "State of Health (%)", "font": {"size": 18, "color": "#8b949e"}},
    ))
    fig_gauge.update_layout(
        height=380,
        template="plotly_dark",
        margin=dict(t=60, b=20, l=40, r=40),
        **LAYOUT_STYLE,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption(
        "🔴 Red threshold line at 80% — below this level, the battery is considered "
        "end-of-life per industry standard (IEC 62660-1)."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if df_raw.empty:
        st.error(
            "🚨 **Dataset not found.**  "
            f"Please place your CSV file inside the `{DATA_DIR}/` folder "
            "and refresh this page."
        )
        st.stop()

    # --- Battery selector (multi-select) ---
    st.markdown(
        "<div class='section-header'>🔬 Dataset Explorer Controls</div>",
        unsafe_allow_html=True,
    )

    battery_ids = []
    if "battery_id" in df_raw.columns:
        all_batteries = sorted(df_raw["battery_id"].unique().tolist())
        battery_ids   = st.multiselect(
            "Select battery cells to display:",
            options=all_batteries,
            default=all_batteries[:4],   # show first 4 by default
            key="battery_select",
        )

    if battery_ids:
        df_plot = df_raw[df_raw["battery_id"].isin(battery_ids)].copy()
    else:
        df_plot = df_raw.copy()

    # Smoothing toggle
    smooth = st.checkbox(
        "Apply rolling-average smoothing (window = 5 cycles)",
        value=True,
        key="smooth_toggle",
    )
    if smooth and "cycle" in df_plot.columns:
        df_plot = df_plot.sort_values(["battery_id", "cycle"])
        for col in ["soh", "capacity"]:
            if col in df_plot.columns:
                df_plot[f"{col}_smooth"] = (
                    df_plot.groupby("battery_id")[col]
                    .transform(lambda s: s.rolling(5, min_periods=1).mean())
                )

    st.divider()

    # -----------------------------------------------------------------------
    #  CHART A — SoH / Capacity degradation over cycles (Line chart)
    # -----------------------------------------------------------------------
    st.markdown(
        "<div class='section-header'>📉 Chart A — Battery Capacity & SoH Degradation Over Cycles</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "This chart illustrates how each battery cell's **State of Health (SoH)** and "
        "**measured capacity (Ah)** decline continuously as the cell ages through charge-discharge cycles. "
        "The red dashed line marks the 80% SoH end-of-life threshold."
    )

    # Decide which y-column to use
    soh_col = "soh_smooth" if (smooth and "soh_smooth" in df_plot.columns) else "soh"
    cap_col = "capacity_smooth" if (smooth and "capacity_smooth" in df_plot.columns) else "capacity"

    col_a1, col_a2 = st.columns(2, gap="medium")

    # ── SoH line chart ──
    with col_a1:
        if "soh" in df_plot.columns and "cycle" in df_plot.columns:
            fig_soh = px.line(
                df_plot,
                x="cycle",
                y=soh_col,
                color="battery_id",
                labels={
                    "cycle":   "Charge Cycle",
                    soh_col:   "State of Health",
                    "battery_id": "Battery Cell",
                },
                title="State of Health (SoH) vs. Cycle Number",
                color_discrete_sequence=px.colors.qualitative.Set2,
                **PX_ARGS,
            )
            # 80% end-of-life threshold line
            fig_soh.add_hline(
                y=0.80,
                line_dash="dash",
                line_color="#f85149",
                line_width=2,
                annotation_text="End-of-Life Threshold (80%)",
                annotation_position="top left",
                annotation_font_color="#f85149",
            )
            fig_soh.update_traces(line=dict(width=2))
            fig_soh.update_layout(
                height=430,
                xaxis_title="Charge Cycle Number",
                yaxis_title="State of Health (0 - 1)",
                yaxis_range=[0.6, 1.05],
                legend_title_text="Battery Cell",
                margin=dict(t=50, b=40, l=50, r=20),
                **LAYOUT_STYLE,
            )
            st.plotly_chart(fig_soh, use_container_width=True)
        else:
            st.warning("Columns `soh` and `cycle` not found in the dataset.")

    # ── Capacity line chart ──
    with col_a2:
        if "capacity" in df_plot.columns and "cycle" in df_plot.columns:
            fig_cap = px.line(
                df_plot,
                x="cycle",
                y=cap_col,
                color="battery_id",
                labels={
                    "cycle":  "Charge Cycle",
                    cap_col:  "Capacity (Ah)",
                    "battery_id": "Battery Cell",
                },
                title="Discharge Capacity (Ah) vs. Cycle Number",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                **PX_ARGS,
            )
            fig_cap.update_traces(line=dict(width=2))
            fig_cap.update_layout(
                height=430,
                xaxis_title="Charge Cycle Number",
                yaxis_title="Capacity (Ah)",
                legend_title_text="Battery Cell",
                margin=dict(t=50, b=40, l=50, r=20),
                **LAYOUT_STYLE,
            )
            st.plotly_chart(fig_cap, use_container_width=True)
        else:
            st.warning("Columns `capacity` and `cycle` not found in the dataset.")

    # ── Combined area chart for SoH fade ──
    if "soh" in df_plot.columns and "cycle" in df_plot.columns and battery_ids:
        st.markdown("**Combined SoH Fade — Area Chart (all selected cells)**")
        fig_area = px.area(
            df_plot,
            x="cycle",
            y=soh_col,
            color="battery_id",
            labels={
                "cycle":   "Charge Cycle",
                soh_col:   "State of Health",
                "battery_id": "Battery Cell",
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
            **PX_ARGS,
        )
        fig_area.add_hline(
            y=0.80,
            line_dash="dot",
            line_color="#f85149",
            line_width=2,
        )
        fig_area.update_layout(
            height=350,
            xaxis_title="Charge Cycle Number",
            yaxis_title="State of Health",
            yaxis_range=[0.55, 1.08],
            margin=dict(t=30, b=40, l=50, r=20),
            **LAYOUT_STYLE,
        )
        st.plotly_chart(fig_area, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    #  CHART B — Temperature vs Cycle  (Scatter + Histogram)
    # -----------------------------------------------------------------------
    st.markdown(
        "<div class='section-header'>🌡️ Chart B — Temperature Behaviour Across Operational Cycles</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Elevated operating temperature accelerates electrolyte degradation and capacity loss. "
        "This pair of charts reveals how **cell temperature evolves** across the battery's lifetime "
        "and how temperature is distributed across the test population."
    )

    col_b1, col_b2 = st.columns(2, gap="medium")

    # ── Scatter: temperature vs cycle ──
    with col_b1:
        if "temperature" in df_plot.columns and "cycle" in df_plot.columns:
            # colour each point by SoH if available, otherwise by battery_id
            if "soh" in df_plot.columns:
                fig_scatter = px.scatter(
                    df_plot,
                    x="cycle",
                    y="temperature",
                    color="soh",
                    color_continuous_scale=["#f85149", "#f0883e", "#3fb950"],
                    range_color=[df_plot["soh"].quantile(0.05),
                                 df_plot["soh"].quantile(0.95)],
                    hover_data=["battery_id", "cycle", "temperature", "soh"],
                    labels={
                        "cycle":       "Charge Cycle",
                        "temperature": "Temperature (°C)",
                        "soh":         "SoH",
                        "battery_id":  "Battery",
                    },
                    title="Temperature vs. Cycle (coloured by SoH)",
                    opacity=0.65,
                    **PX_ARGS,
                )
                fig_scatter.update_coloraxes(
                    colorbar_title="SoH",
                    colorbar_tickformat=".2f",
                )
            else:
                fig_scatter = px.scatter(
                    df_plot,
                    x="cycle",
                    y="temperature",
                    color="battery_id",
                    labels={
                        "cycle":       "Charge Cycle",
                        "temperature": "Temperature (°C)",
                        "battery_id":  "Battery",
                    },
                    title="Temperature vs. Cycle Number",
                    opacity=0.65,
                    **PX_ARGS,
                )

            # Trend line using a rolling mean per battery
            if battery_ids:
                trend_frames = []
                for bid in battery_ids:
                    sub = df_plot[df_plot["battery_id"] == bid].sort_values("cycle")
                    sub = sub.assign(temp_trend=sub["temperature"].rolling(10, min_periods=1).mean())
                    trend_frames.append(sub)
                df_trend = pd.concat(trend_frames)
                for bid in battery_ids:
                    s = df_trend[df_trend["battery_id"] == bid]
                    fig_scatter.add_trace(go.Scatter(
                        x=s["cycle"],
                        y=s["temp_trend"],
                        mode="lines",
                        line=dict(width=2, dash="dash"),
                        name=f"{bid} trend",
                        showlegend=False,
                    ))

            fig_scatter.update_traces(marker=dict(size=5))
            fig_scatter.update_layout(
                height=430,
                xaxis_title="Charge Cycle Number",
                yaxis_title="Temperature (°C)",
                margin=dict(t=50, b=40, l=50, r=20),
                **LAYOUT_STYLE,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Columns `temperature` and `cycle` not found in the dataset.")

    # ── Histogram: temperature distribution ──
    with col_b2:
        if "temperature" in df_plot.columns:
            fig_hist = px.histogram(
                df_plot,
                x="temperature",
                color="battery_id" if "battery_id" in df_plot.columns else None,
                nbins=50,
                opacity=0.8,
                barmode="overlay",
                labels={
                    "temperature": "Temperature (°C)",
                    "battery_id":  "Battery Cell",
                },
                title="Temperature Distribution Across All Cycles",
                color_discrete_sequence=px.colors.qualitative.Set2,
                **PX_ARGS,
            )
            fig_hist.update_layout(
                height=430,
                xaxis_title="Temperature (°C)",
                yaxis_title="Count",
                legend_title_text="Battery Cell",
                bargap=0.05,
                margin=dict(t=50, b=40, l=50, r=20),
                **LAYOUT_STYLE,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Column `temperature` not found in the dataset.")

    # ── Temperature × Cycle box-plot ──
    st.divider()
    if "temperature" in df_plot.columns and "cycle" in df_plot.columns:
        st.markdown("**Temperature spread per cycle bin (Box Plot)**")

        # Bin cycles into 10-cycle buckets for readability
        df_plot["cycle_bin"] = (df_plot["cycle"] // 10) * 10
        fig_box = px.box(
            df_plot,
            x="cycle_bin",
            y="temperature",
            color="battery_id" if "battery_id" in df_plot.columns else None,
            labels={
                "cycle_bin":   "Cycle (binned by 10)",
                "temperature": "Temperature (°C)",
                "battery_id":  "Battery Cell",
            },
            title="Temperature Spread per Cycle Bin",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            **PX_ARGS,
        )
        fig_box.update_layout(
            height=380,
            xaxis_title="Cycle Bucket (every 10 cycles)",
            yaxis_title="Temperature (°C)",
            margin=dict(t=50, b=40, l=50, r=20),
            **LAYOUT_STYLE,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heat-map ──
    st.divider()
    numeric_cols = [c for c in ["cycle", "voltage", "temperature", "capacity", "soh", "rul"]
                    if c in df_plot.columns]
    if len(numeric_cols) >= 2:
        st.markdown("**Feature Correlation Matrix**")
        corr = df_plot[numeric_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0,  "#f85149"],
                [0.5,  "#161b22"],
                [1.0,  "#3fb950"],
            ],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            hoverongaps=False,
        ))
        fig_corr.update_layout(
            height=420,
            template="plotly_dark",
            margin=dict(t=40, b=40, l=60, r=20),
            **LAYOUT_STYLE,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "Green = strong positive correlation · Red = strong negative correlation. "
            "Cycle count has a strong negative correlation with SoH, confirming expected degradation."
        )

    # ── Raw data table (expandable) ──
    with st.expander("📋 View raw dataset (first 200 rows)"):
        st.dataframe(
            df_plot.drop(
                columns=[c for c in df_plot.columns if c.endswith("_smooth") or c == "cycle_bin"],
                errors="ignore",
            ).head(200),
            use_container_width=True,
        )