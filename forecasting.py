import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. ENTERPRISE UI CONFIGURATION ---
st.set_page_config(page_title="Supply Chain AI | Precision Forecast", layout="wide", initial_sidebar_state="collapsed")

# Custom Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Base Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #1E293B;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    /* Professional Header */
    .main-header {
        text-align: center;
        padding: 60px 0 40px 0;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 40px;
    }
    
    .main-header h1 {
        font-weight: 800;
        letter-spacing: -1px;
        color: #0F172A;
        font-size: 2.8rem;
    }

    /* Vertical Roadmap Step */
    .step-container {
        max-width: 850px;
        margin: 0 auto 30px auto;
        padding: 30px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        transition: all 0.2s ease-in-out;
    }
    
    .step-container:hover {
        border-color: #3B82F6;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05);
    }

    .step-tag {
        font-size: 0.7rem;
        font-weight: 700;
        color: #3B82F6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        display: block;
    }

    .step-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 20px;
    }

    /* Execute Button - Enterprise Style */
    div.stButton > button {
        width: 100%;
        max-width: 850px;
        display: block;
        margin: 40px auto;
        background-color: #0F172A !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #3B82F6 !important;
        transform: translateY(-2px);
    }

    /* Report Section */
    .report-card {
        background: #F8FAFC;
        border-radius: 20px;
        padding: 40px;
        margin-top: 50px;
        border: 1px solid #E2E8F0;
    }

    /* Hide standard Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('''
<div class="main-header">
    <span style="color: #3B82F6; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">Demand Intelligence Platform</span>
    <h1>AI Precision Forecast</h1>
    <p style="color: #64748B; font-size: 1.2rem; max-width: 600px; margin: 10px auto;">Advanced supply chain optimization powered by machine learning and statistical modeling.</p>
</div>
''', unsafe_allow_html=True)

# --- STEP 1: SCOPE ---
st.markdown('<div class="step-container"><span class="step-tag">Phase 01</span><div class="step-title">Forecasting Scope Definition</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_choice = st.radio("Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with c2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Granularity", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: TIMELINE ---
st.markdown('<div class="step-container"><span class="step-tag">Phase 02</span><div class="step-title">Temporal Parameters</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    interval = st.selectbox("Data Resolution", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c4:
    horizon_label = st.selectbox("Target Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: STRATEGY ---
st.markdown('<div class="step-container"><span class="step-tag">Phase 03</span><div class="step-title">Statistical Strategy Selection</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c6:
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weight Distribution", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback Window (Period)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Multiplier", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-container"><span class="step-tag">Phase 04</span><div class="step-title">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Historical Demand (CSV or Excel)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CALCULATION LOGIC ---
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
            selected = st.selectbox("🎯 Target Entity Selection", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)
            st.markdown('</div>', unsafe_allow_html=True)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("GENERATE AI ANALYTICS REPORT"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#0F172A; margin-bottom:30px;'>Forecast Report: {item_name}</h2>", unsafe_allow_html=True)
            
            # Prediction Logic
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            target_df['month'], target_df['dow'] = target_df['Date'].dt.month, target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base_scalar
            
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            last_date, last_qty = target_df['Date'].max(), target_df['qty'].iloc[-1]
            h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
            end_date = last_date + pd.Timedelta(days=h_map[horizon_label])
            
            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            excel_calc_col, predicted_calc_col = [], []
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # CHART
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Historical Demand", line=dict(color="#0F172A", width=3)))
            
            f_dates_conn = [last_date] + list(future_dates)
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+excel_calc_col, name="Baseline (Stat)", line=dict(color="#94A3B8", dash='dot')))
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+predicted_calc_col, name="AI Prediction", line=dict(color="#3B82F6", width=4)))
            
            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(l=0,r=0,t=20,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # DATA TABLE
            st.markdown("<br>### Predicted Schedule", unsafe_allow_html=True)
            res_df = pd.DataFrame({"Date": future_dates.strftime('%Y-%m-%d'), "AI Forecast": predicted_calc_col, "Statistical Baseline": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            # EXPORT
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("📥 EXPORT DATA TO EXCEL", output.getvalue(), f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
