import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. PREMIUM ENTERPRISE UI CONFIG ---
st.set_page_config(page_title="DemandIntel AI | Precision Forecast", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for high-end SaaS aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #111827;
    }

    /* Hide standard Streamlit elements */
    header, footer, #MainMenu {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Hero Section Styling */
    .hero-text h1 {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -2.5px;
        line-height: 1;
        color: #0F172A;
        margin-bottom: 20px;
    }
    
    .hero-text span {
        color: #00D1FF;
    }

    .hero-subtitle {
        color: #64748B;
        font-size: 1.25rem;
        max-width: 500px;
        line-height: 1.6;
    }

    /* Vertical Roadmap Design */
    .step-wrapper {
        position: relative;
        padding-left: 45px;
        margin-bottom: 50px;
        border-left: 2px solid #F1F5F9;
    }

    .step-dot {
        position: absolute;
        left: -9px;
        top: 0;
        width: 16px;
        height: 16px;
        background-color: #FFFFFF;
        border: 3px solid #00D1FF;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(0, 209, 255, 0.4);
    }

    .step-badge {
        font-size: 0.7rem;
        font-weight: 700;
        color: #00B4D8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        display: block;
    }

    .step-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 20px;
    }

    /* Professional Button */
    div.stButton > button {
        width: 100% !important;
        background: linear-gradient(90deg, #0F172A 0%, #334155 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 18px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-top: 20px;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #00D1FF 0%, #0077B6 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 209, 255, 0.3);
    }

    /* Result Card */
    .insight-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 40px;
        border-radius: 24px;
        margin-top: 40px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.05);
    }
    
    /* Input Styling */
    .stSelectbox label, .stRadio label {
        font-weight: 600 !important;
        color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
col_h1, col_h2 = st.columns([1.3, 1])

with col_h1:
    st.markdown("""
        <div class="hero-text" style="margin-top: 50px;">
            <h1 style="font-size: 3rem; font-weight: 800; letter-spacing: -2px; color: #4F46E5;">Agilo<span style="color:#111827;">Forecast</span></h1>
            <p class="hero-subtitle">High-precision AI supply chain forecasting engine built for modern enterprise demand planning.</p>
        </div>
    """, unsafe_allow_html=True)

with col_h2:
    # Adding the provided teal forecast image
    st.image("forecast.png", use_column_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- VERTICAL ROADMAP FLOW ---

# STEP 1
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<span class="step-badge">Phase 01</span><div class="step-title">Forecasting Scope</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
with c2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Resolution Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# STEP 2
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<span class="step-badge">Phase 02</span><div class="step-title">Timeline Parameters</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c4:
    horizon_label = st.selectbox("Default Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# STEP 3
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<span class="step-badge">Phase 03</span><div class="step-title">Modeling Strategy</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c6:
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weights", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback Window", 2, 30, 7)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# STEP 4
st.markdown('<div class="step-wrapper" style="border-left: none;"><div class="step-dot"></div>'
            '<span class="step-badge">Phase 04</span><div class="step-title">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload ERP / Excel Historical Data", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE CALCULATION LOGIC ---
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
            item_name = "System Aggregate"
        else:
            selected = st.selectbox("Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("🚀 EXECUTE PREDICTIVE ANALYSIS"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            
            # Dynamic Horizon Control
            st.markdown("### 🛠 Operational Adjustment")
            cx1, cx2 = st.columns(2)
            with cx1: dynamic_val = st.number_input("Lookahead Length", min_value=1, value=15)
            with cx2: dynamic_unit = st.selectbox("Time Unit", ["Days", "Weeks", "Months", "Original Selection"])
            
            # AI Logic
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
                base = excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- PRECISE CURVY CHART AS REQUESTED ---
            st.subheader(f"📈 Trajectory Forecast: {item_name}")
            fig = go.Figure()

            # Traded (Actuals)
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Historical Traded",
                mode='lines+markers', line=dict(color="#1a8cff", width=2.5, shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#1a8cff", width=1.5))
            ))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            # Baseline
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Baseline Forecast",
                mode='lines+markers', line=dict(color="#94a3b8", width=1.2, dash='dot', shape='spline'),
                marker=dict(size=4, color="#94a3b8")
            ))

            # AI Forecast
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Prediction",
                mode='lines+markers', line=dict(color="#00D1FF", width=3, dash='dash', shape='spline'),
                marker=dict(size=5, color="white", line=dict(color="#00D1FF", width=1.5))
            ))

            fig.add_vline(x=last_date, line_width=1.5, line_color="#cccccc")
            
            # Historical Icon
            fig.add_annotation(x=target_df['Date'].iloc[int(len(target_df)*0.8)], y=target_df['qty'].max()*1.1, text="🛍️", showarrow=False, bgcolor="rgba(26,140,255,0.1)", bordercolor="#1a8cff", borderwidth=1.5, borderpad=6)
            
            # Forecast Icon
            if len(future_dates) > 0:
                fig.add_annotation(x=future_dates[int(len(future_dates)*0.5)], y=max(predicted_calc_col)*1.1, text="📢", showarrow=False, bgcolor="rgba(0,209,255,0.1)", bordercolor="#00D1FF", borderwidth=1.5, borderpad=6)

            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # --- AI WIGGLE CHART ---
            st.subheader("📉 AI Pattern Variance (The Wiggles)")
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Adjustment", marker_color="#00D1FF"
            ))
            fig_wig.update_layout(template="plotly_white", height=300, title="Machine learning adjustment to the baseline")
            st.plotly_chart(fig_wig, use_container_width=True)

            # DATA EXPORT
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Forecast": predicted_calc_col, "Baseline": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("📥 DOWNLOAD ENTERPRISE REPORT", output.getvalue(), f"Forecast_Report_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"System Error: {e}")
