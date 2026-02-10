import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. PREMIUM UI & CLEAN WHITE CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }

    .stApp {
        background-color: #ffffff;
    }

    /* Main Container */
    .main-content {
        max-width: 900px;
        margin: 0 auto;
    }

    /* Step Card Styling */
    .step-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }

    .step-container:hover {
        border-color: #2563eb;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* Number Badge */
    .step-badge {
        display: inline-block;
        background: #2563eb;
        color: white;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 12px;
    }

    .step-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 20px;
    }

    /* Input Field Styling */
    .stSelectbox, .stRadio, .stNumberInput {
        margin-bottom: 10px;
    }

    /* Execute Button */
    div.stButton > button {
        width: 100%;
        background-color: #111827 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 16px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: none !important;
        margin-top: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    div.stButton > button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px);
    }

    /* Dynamic Horizon Box */
    .control-panel {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 30px;
    }

    /* Table Styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<div style="text-align: center; padding: 40px 0;">'
            '<h1 style="font-size: 2.5rem; color: #111827; font-weight: 800; margin-bottom: 10px;">📊 AI Supply Chain Precision</h1>'
            '<p style="color: #6b7280; font-size: 1.1rem;">Follow the vertical roadmap to generate your automated forecasts.</p>'
            '</div>', unsafe_allow_html=True)

# Wrap everything in a centered div
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# --- STEP 1: SCOPE ---
st.markdown('<div class="step-container"><span class="step-badge">STEP 1</span><div class="step-title">Forecasting Scope</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
with c2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Detail Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: TIMELINE ---
st.markdown('<div class="step-container"><span class="step-badge">STEP 2</span><div class="step-title">Timeline Configuration</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c4:
    horizon_label = st.selectbox("Default Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: STRATEGY ---
st.markdown('<div class="step-container"><span class="step-badge">STEP 3</span><div class="step-title">Select Baseline Technique</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    technique = st.selectbox("Statistical Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c6:
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Weights (comma separated)", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback Window (n)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing (Alpha)", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-container"><span class="step-badge">STEP 4</span><div class="step-title">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel file (Dates as columns)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE CALCULATION LOGIC ---
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

# --- EXECUTION FLOW ---
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
            item_name = "Aggregate Total"
        else:
            selected = st.selectbox("Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("🚀 EXECUTE PREDICTIVE ENGINE"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<hr style="margin: 50px 0; border-top: 2px solid #f3f4f6;">', unsafe_allow_html=True)
            
            # --- RESULTS SECTION ---
            st.markdown('<div class="control-panel"><b>🔄 Dynamic Horizon Adjustment</b><br><br>', unsafe_allow_html=True)
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                dynamic_val = st.number_input("Prediction Length", min_value=1, value=15)
            with c_h2:
                dynamic_unit = st.selectbox("Time Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculation Core
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
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

            # --- PLOTTING ---
            st.subheader(f"📈 Trend Analysis: {item_name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Actual History", line=dict(color="#111827", width=3, shape='spline')))
            
            f_dates_conn = [last_date] + list(future_dates)
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+excel_calc_col, name="Baseline", line=dict(color="#94a3b8", width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+predicted_calc_col, name="AI Prediction", line=dict(color="#2563eb", width=4, shape='spline')))
            
            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(l=0,r=0,t=40,b=0), 
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # AI Pattern Bar
            st.subheader("📉 AI Correction Patterns (Residuals)")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, marker_color="#3b82f6"))
            fig_wig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_wig, use_container_width=True)

            # Table & Export
            st.subheader("📋 Forecasted Output")
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Predicted Qty": predicted_calc_col, "Excel Baseline": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("📥 Download Full Prediction (Excel)", output.getvalue(), f"AI_Forecast_{item_name}.xlsx")

    except Exception as e:
        st.error(f"Processing Error: {e}")

st.markdown('</div>', unsafe_allow_html=True) # End Main Content
