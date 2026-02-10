import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. ENTERPRISE SaaS UI CONFIG ---
st.set_page_config(page_title="DemandIntel | AI Supply Chain", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for high-end SaaS aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    /* Global Body styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #F8FAFC;
        color: #1E293B;
    }

    /* Hide standard Streamlit header/footer */
    header, footer, #MainMenu {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Hero Section */
    .hero-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .hero-text h1 {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -2px;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }
    
    .hero-text span {
        color: #00D1FF; /* Teal Glow */
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748B;
        max-width: 500px;
        line-height: 1.6;
    }

    /* Modern Phase Cards */
    .phase-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 6px solid #00D1FF;
    }

    .phase-badge {
        background: rgba(0, 209, 255, 0.1);
        color: #00B4D8;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        display: inline-block;
    }

    .phase-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1.5rem;
    }

    /* Executive Execute Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0F172A 0%, #334155 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 1.2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-top: 2rem;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #00D1FF 0%, #0077B6 100%) !important;
        box-shadow: 0 20px 25px -5px rgba(0, 209, 255, 0.3);
    }

    /* Result Analytics Box */
    .analytics-container {
        background: #FFFFFF;
        border-radius: 30px;
        padding: 3rem;
        margin-top: 4rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.05);
    }

    /* Style for file uploader */
    .stFileUploader section {
        border-radius: 15px !important;
        border: 2px dashed #CBD5E1 !important;
    }

</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
col_h1, col_h2 = st.columns([1.3, 1])

with col_h1:
    st.markdown("""
        <div class="hero-text">
            <h1>DemandIntel<span>.ai</span></h1>
            <p class="hero-subtitle">The next-generation AI forecasting engine designed for high-precision supply chain management.</p>
        </div>
    """, unsafe_allow_html=True)

with col_h2:
    try:
        # Use your "Order Forecast" image here
        st.image("forecast_image.png", use_column_width=True)
    except:
        # Fallback Gradient if image is not found
        st.markdown('<div style="background: linear-gradient(135deg, #F0FDFF 0%, #CCF5FF 100%); height: 350px; border-radius: 30px;"></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- ROADMAP CARDS ---

# Phase 1
st.markdown('<div class="phase-card"><div class="phase-badge">Phase 01</div><div class="phase-title">Forecasting Scope</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_choice = st.radio("Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with c2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Granularity", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# Phase 2 & 3 Combined for sleekness
st.markdown('<div class="phase-card"><div class="phase-badge">Phase 02 & 03</div><div class="phase-title">Temporal & Logic Logic</div>', unsafe_allow_html=True)
c3, c4, c5 = st.columns(3)
with c3:
    interval = st.selectbox("Interval Frequency", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c4:
    horizon_label = st.selectbox("Target Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
with c5:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
st.markdown('</div>', unsafe_allow_html=True)

# Phase 4
st.markdown('<div class="phase-card"><div class="phase-badge">Phase 04</div><div class="phase-title">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload ERP / Historical Data (CSV or Excel)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC ---
def calculate_excel_baseline(demand, tech):
    if len(demand) == 0: return 0
    return np.mean(demand[-7:]) if tech == "Moving Average" else np.mean(demand)

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
            st.markdown("### Select Specific Target")
            selected = st.selectbox("🎯 Identity Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # Action Button
        if st.button("🚀 EXECUTE PREDICTIVE ENGINE"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color:#0F172A;'>Analytical Intelligence Report: {item_name}</h2><br>", unsafe_allow_html=True)
            
            # Prediction
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique)
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

            # --- PRECISE CURVY CHART ---
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Historical", mode='lines+markers', line=dict(color="#0F172A", width=3, shape='spline')))
            # Baseline
            f_dates_conn = [last_date] + list(future_dates)
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+excel_calc_col, name="Stat-Baseline", mode='lines', line=dict(color="#94A3B8", dash='dot')))
            # AI
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+predicted_calc_col, name="AI Forecast", mode='lines+markers', line=dict(color="#00D1FF", width=4, shape='spline')))
            
            fig.update_layout(template="plotly_white", hovermode="x unified", height=550, margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # --- SEASONALITY ADJUSTMENT ---
            st.markdown("### AI Seasonality Logic (The Wiggles)")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, marker_color="#00D1FF"))
            fig_wig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig_wig, use_container_width=True)

            # EXPORT
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Forecast": predicted_calc_col})
            st.dataframe(res_df, use_container_width=True)
            st.download_button("📥 DOWNLOAD ENTERPRISE REPORT", "data", "Forecast_Report.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"System Error: {e}")
