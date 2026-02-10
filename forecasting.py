import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. PREMIUM SaaS UI CONFIG ---
st.set_page_config(page_title="DemandIntel AI", layout="wide", initial_sidebar_state="collapsed")

# Professional CSS for "Enterprise SaaS" Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #FFFFFF;
        color: #1A202C;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }

    /* Header Styling */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -3px;
        line-height: 0.9;
        margin-bottom: 20px;
        color: #0F172A;
    }
    
    .hero-subtitle {
        color: #64748B;
        font-size: 1.2rem;
        font-weight: 400;
        max-width: 500px;
        line-height: 1.6;
    }

    /* Modern Vertical Timeline */
    .step-wrapper {
        position: relative;
        padding-left: 50px;
        margin-bottom: 60px;
        border-left: 1.5px solid #E2E8F0;
    }

    .step-dot {
        position: absolute;
        left: -10px;
        top: 0;
        width: 18px;
        height: 18px;
        background-color: #FFFFFF;
        border: 3px solid #00D1FF;
        border-radius: 50%;
        box-shadow: 0 0 15px rgba(0, 209, 255, 0.4);
    }

    .step-label {
        color: #00D1FF;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .step-heading {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 25px;
    }

    /* Customizing Streamlit Widgets to look 'Paid' */
    div[data-baseweb="select"] {
        border-radius: 12px !important;
        background-color: #F8FAFC !important;
    }
    
    .stRadio div[role="radiogroup"] {
        background: #F8FAFC;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #F1F5F9;
    }

    /* The "Execute" Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0F172A 0%, #334155 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 20px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        background: #00D1FF !important;
        box-shadow: 0 15px 30px -5px rgba(0, 209, 255, 0.3);
    }

    /* Dashboard Result Card */
    .dashboard-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 40px;
        border-radius: 30px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.03);
        margin-top: 50px;
    }

    hr { border-top: 1px solid #F1F5F9; margin: 40px 0; }

</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_h1, col_h2 = st.columns([1.3, 1])

with col_h1:
    st.markdown('<div style="margin-top: 40px;">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">DemandIntel<span style="color:#00D1FF;">.ai</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">The next generation AI forecasting engine designed for high-precision supply chain management.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_h2:
    # Attempting to load the image if present
    try:
        st.image("forecast_image.png", use_column_width=True) 
    except:
        st.markdown('<div style="background: linear-gradient(135deg, #F0FDFF 0%, #CCF5FF 100%); height: 350px; border-radius: 30px; border: 1px solid #E0F2FE;"></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- STEP 1 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 01</div><div class="step-heading">Forecasting Scope</div>', unsafe_allow_html=True)
col_s1_a, col_s1_b = st.columns(2)
with col_s1_a:
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col_s1_b:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Resolution Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 02</div><div class="step-heading">Time Parameters</div>', unsafe_allow_html=True)
col_s2_a, col_s2_b = st.columns(2)
with col_s2_a:
    interval = st.selectbox("Frequency", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_s2_b:
    horizon_label = st.selectbox("Target Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 03</div><div class="step-heading">Logic Strategy</div>', unsafe_allow_html=True)
col_s3_a, col_s3_b = st.columns(2)
with col_s3_a:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with col_s3_b:
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weight Ratios", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Window", 2, 30, 7)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4 ---
st.markdown('<div class="step-wrapper" style="border-left:none;"><div class="step-dot"></div>'
            '<div class="step-label">Step 04</div><div class="step-heading">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload ERP / Excel Data", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC ---
def calculate_excel_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.33, 0.33, 0.34]))
        n = len(w)
        return np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
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
            item_name = "Global Aggregate"
        else:
            selected = st.selectbox("Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("RUN PREDICTIVE ENGINE"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            
            # Prediction
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
                base = excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # CHART
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="History", line=dict(color="#0F172A", width=3, shape='spline')))
            f_dates_conn = [last_date] + list(future_dates)
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+excel_calc_col, name="Stat-Baseline", line=dict(color="#94A3B8", dash='dot')))
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+predicted_calc_col, name="AI Forecast", line=dict(color="#00D1FF", width=4, shape='spline')))
            
            fig.update_layout(template="plotly_white", height=500, margin=dict(l=0,r=0,t=20,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # DATA
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Forecast": predicted_calc_col})
            st.dataframe(res_df, use_container_width=True)
            
            st.download_button("📥 DOWNLOAD REPORT", "data", "Report.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
