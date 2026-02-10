import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. ENTERPRISE UI CONFIG ---
st.set_page_config(page_title="AI Supply Chain Precision", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    /* Minimalist Step Headers */
    .step-badge {
        font-size: 0.7rem;
        font-weight: 700;
        color: #6366F1;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 5px;
        display: block;
    }

    .step-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 25px;
    }

    /* Vertical Flow Container */
    .vertical-step {
        max-width: 850px;
        margin: 0 auto;
        padding: 40px 0;
        border-bottom: 1px solid #F3F4F6;
    }

    /* Large Execute Button */
    div.stButton > button {
        width: 100% !important;
        max-width: 850px;
        display: block;
        margin: 40px auto !important;
        background-color: #111827 !important;
        color: white !important;
        padding: 20px !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #6366F1 !important;
        transform: translateY(-2px);
    }

    /* Dynamic Horizon Box */
    .adjustment-box {
        background-color: #F9FAFB;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #E5E7EB;
        margin-bottom: 30px;
    }

</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div style="text-align: center; padding: 60px 0 20px 0;">'
            '<h1 style="font-size: 3rem; font-weight: 800; color: #111827; letter-spacing: -1.5px;">Precision Forecast<span style="color:#6366F1;">.ai</span></h1>'
            '<p style="color: #6B7280; font-size: 1.2rem;">Enterprise-grade demand intelligence and supply chain optimization.</p>'
            '</div>', unsafe_allow_html=True)

# --- VERTICAL STEPS ---

# STEP 1
st.markdown('<div class="vertical-step"><span class="step-badge">Phase 01</span><div class="step-title">Forecasting Scope</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Primary Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Resolution Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# STEP 2
st.markdown('<div class="vertical-step"><span class="step-badge">Phase 02</span><div class="step-title">Timeline Configuration</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    horizon_label = st.selectbox("Default Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# STEP 3
st.markdown('<div class="vertical-step"><span class="step-badge">Phase 03</span><div class="step-title">Strategy & AI Technique</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c4:
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weights (comma separated)", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# STEP 4
st.markdown('<div class="vertical-step"><span class="step-badge">Phase 04</span><div class="step-title">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel (Horizontal Format)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE LOGIC ---
def calculate_excel_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.5, 0.5]))
        n = len(w)
        return np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
    elif tech == "Ramp Up Evenly":
        d_slice = demand[-7:] 
        weights = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, weights) / weights.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

# --- EXECUTION ---
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "System Aggregate"
        else:
            st.markdown('<div style="max-width:850px; margin: 0 auto;">', unsafe_allow_html=True)
            selected = st.selectbox("🎯 Target Identification", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)
            st.markdown('</div>', unsafe_allow_html=True)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("🚀 EXECUTE AI TREND ANALYSIS"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            # --- RESULTS SECTION ---
            st.markdown('<div style="max-width:1000px; margin: 40px auto;">', unsafe_allow_html=True)
            
            # Adjustment viewport
            st.markdown('<div class="adjustment-box"><b>🛠 Operational Viewport Adjustment</b>', unsafe_allow_html=True)
            ax1, ax2 = st.columns(2)
            with ax1: dynamic_val = st.number_input("Lookahead Length", min_value=1, value=15)
            with ax2: dynamic_unit = st.selectbox("Time Metric", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Core Modeling
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            target_df['month'], target_df['dow'] = target_df['Date'].dt.month, target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base_scalar
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            last_date, last_qty = target_df['Date'].max(), target_df['qty'].iloc[-1]
            if dynamic_unit == "Original Selection":
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
                end_date = last_date + pd.Timedelta(days=h_map[horizon_label])
            elif dynamic_unit == "Days": end_date = last_date + pd.Timedelta(days=dynamic_val)
            elif dynamic_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=dynamic_val)
            else: end_date = last_date + pd.DateOffset(months=dynamic_val)
            
            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            excel_calc_col, predicted_calc_col = [], []
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- PRECISE GRAPH CODE AS REQUESTED ---
            st.subheader(f"📈 Predictive Trend Analysis: {item_name}")
            fig = go.Figure()

            # TRADED
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Traded",
                mode='lines+markers', line=dict(color="#1a8cff", width=2.5, shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#1a8cff", width=1.5))
            ))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            # EXCEL BASELINE
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Excel Calculated Forecast",
                mode='lines+markers', line=dict(color="#999999", width=1.2, dash='dot', shape='spline'),
                marker=dict(size=4, color="#999999")
            ))

            # AI PREDICTION
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Predicted Forecast",
                mode='lines+markers', line=dict(color="#ffcc00", width=2.5, dash='dash', shape='spline'),
                marker=dict(size=5, color="white", line=dict(color="#ffcc00", width=1.5))
            ))

            fig.add_vline(x=last_date, line_width=1.5, line_color="#cccccc")
            fig.add_annotation(x=target_df['Date'].iloc[int(len(target_df)*0.8)], y=target_df['qty'].max()*1.1, text="🛍️", showarrow=False, bgcolor="rgba(26,140,255,0.1)", bordercolor="#1a8cff", borderwidth=1.5, borderpad=6)
            fig.add_annotation(x=future_dates[int(len(future_dates)*0.5)] if len(future_dates)>0 else last_date, y=max(predicted_calc_col)*1.1 if len(predicted_calc_col)>0 else last_qty, text="📢", showarrow=False, bgcolor="rgba(255,204,0,0.1)", bordercolor="#ffcc00", borderwidth=1.5, borderpad=6)

            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # --- AI WIGGLE CHART ---
            st.subheader("📉 AI Pattern Adjustment (The Wiggles)")
            st.info("This chart shows exactly how much the AI is adding or subtracting from the Excel baseline based on detected patterns.")
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Adjustment", marker_color="#00B0F0"
            ))
            fig_wig.update_layout(template="plotly_white", height=300, title="Negative/Positive Patterns identified by AI")
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- TABLE ---
            st.subheader("📋 Forecast Output Table")
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Forecast": predicted_calc_col, "Statistical Baseline": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("📥 DOWNLOAD REPORT", output.getvalue(), f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Critical System Error: {e}")
