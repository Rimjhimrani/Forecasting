import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main { 
        background: #0f0f1e;
        font-family: 'Poppins', sans-serif;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0 3rem 3rem 3rem !important;
        max-width: 1400px !important;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 0;
        margin: 0 -3rem 3rem -3rem;
        text-align: center;
        border-bottom: 4px solid #a855f7;
    }
    
    .hero-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 4px 8px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Section Dividers */
    .section-divider {
        display: flex;
        align-items: center;
        margin: 3rem 0 2rem 0;
        position: relative;
    }
    
    .section-number {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-right: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5);
        flex-shrink: 0;
    }
    
    .section-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        flex-grow: 1;
    }
    
    .section-line {
        height: 2px;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.3), transparent);
        flex-grow: 1;
        margin-left: 1.5rem;
    }
    
    /* Input Groups */
    .input-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    /* Labels */
    label {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background-color: #1a1a2e !important;
        border: 2px solid #2d2d44 !important;
        border-radius: 12px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #2d2d44;
    }
    
    .stRadio > div > label {
        display: flex;
        gap: 1rem;
    }
    
    .stRadio label {
        color: #e0e0e0 !important;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background-color: #1a1a2e !important;
        border: 2px solid #2d2d44 !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background-color: #1a1a2e !important;
        border: 2px solid #2d2d44 !important;
        border-radius: 12px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #2d2d44 !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #667eea !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%) !important;
        padding: 2.5rem !important;
        border-radius: 16px !important;
        border: 2px dashed #667eea !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #a855f7 !important;
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.3) !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #e0e0e0 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
    }
    
    /* Execute Button */
    .execute-section {
        margin: 3rem 0;
        text-align: center;
    }
    
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1.5rem 3rem !important;
        border-radius: 16px !important;
        border: none !important;
        font-size: 1.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 50px rgba(16, 185, 129, 0.5) !important;
    }
    
    /* Results Section */
    .results-header {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        padding: 2rem;
        margin: 3rem -3rem 2rem -3rem;
        text-align: center;
        border-top: 4px solid #a855f7;
        border-bottom: 4px solid #ec4899;
    }
    
    .results-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    }
    
    /* Control Panel */
    .control-panel {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(251, 191, 36, 0.3);
    }
    
    .control-title {
        color: #78350f;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Chart Container */
    .chart-section {
        background: #16213e;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid #2d2d44;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .chart-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Info Box */
    .stAlert {
        background-color: #1e3a5f !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    
    /* DataFrame */
    .dataframe {
        background-color: #1a1a2e !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stDataFrame"] {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        border: none !important;
        font-size: 1.1rem !important;
        box-shadow: 0 6px 24px rgba(236, 72, 153, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 32px rgba(236, 72, 153, 0.5) !important;
    }
    
    /* Divider */
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📊 AI Precision Forecast</div>
        <div class="hero-subtitle">Intelligent Supply Chain Demand Forecasting with Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

# --- SECTION 1: SCOPE ---
st.markdown("""
    <div class="section-divider">
        <div class="section-number">1</div>
        <div class="section-title">Forecasting Scope Selection</div>
        <div class="section-line"></div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("🎯 Primary Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("🔍 Specific Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- SECTION 2: TIMELINE ---
st.markdown("""
    <div class="section-divider">
        <div class="section-number">2</div>
        <div class="section-title">Timeline Configuration</div>
        <div class="section-line"></div>
    </div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    interval = st.selectbox("⏱️ Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_b:
    horizon_label = st.selectbox("📅 Default Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)

# --- SECTION 3: TECHNIQUES ---
st.markdown("""
    <div class="section-divider">
        <div class="section-number">3</div>
        <div class="section-title">Select Strategy & AI Technique</div>
        <div class="section-line"></div>
    </div>
""", unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    technique = st.selectbox("🧮 Excel Strategy (Baseline)", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
with col_d:
    if technique == "Weightage Average":
        w_in = st.text_input("⚖️ Manual Weights (comma separated)", "0.2, 0.3, 0.5")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("🔄 Lookback Window (n)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("📈 Growth Factor (Multiplier)", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("🎚️ Smoothing Alpha", 0.01, 1.0, 0.3)

# --- SECTION 4: UPLOAD ---
st.markdown("""
    <div class="section-divider">
        <div class="section-number">4</div>
        <div class="section-title">Data Ingestion</div>
        <div class="section-line"></div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload CSV/Excel (Dates as Columns)", type=['xlsx', 'csv'])

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
            item_name = "Aggregate Sum"
        else:
            selected = st.selectbox(f"🔎 Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="execute-section">', unsafe_allow_html=True)
        if st.button("🚀 EXECUTE AI TREND ANALYSIS"):
            st.session_state.run_analysis = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('run_analysis', False):
            
            # --- RESULTS HEADER ---
            st.markdown("""
                <div class="results-header">
                    <div class="results-title">⚡ AI Analysis Results</div>
                </div>
            """, unsafe_allow_html=True)
            
            # --- CONTROL PANEL ---
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown('<div class="control-title">🔄 Adjust Forecast Horizon Dynamically</div>', unsafe_allow_html=True)
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("📊 Enter Quantity", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("📐 Select Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # 1. Excel Baseline scalar
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            # 2. AI Model Training (Residuals)
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base_scalar
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            # 3. Future Dates Calculation
            last_date = target_df['Date'].max()
            last_qty = target_df['qty'].iloc[-1]
            
            if dynamic_unit == "Original Selection":
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
                end_date = last_date + pd.Timedelta(days=h_map[horizon_label])
            elif dynamic_unit == "Days": end_date = last_date + pd.Timedelta(days=dynamic_val)
            elif dynamic_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=dynamic_val)
            else: end_date = last_date + pd.DateOffset(months=dynamic_val)
            
            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            
            # 4. Predictions Construction
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            excel_calc_col = []
            predicted_calc_col = []
            
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- TREND GRAPH ---
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-title">📈 Predictive Trend Analysis: {item_name}</div>', unsafe_allow_html=True)
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

            fig.update_layout(
                template="plotly_dark", 
                hovermode="x unified", 
                height=500, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- AI WIGGLE CHART ---
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">📉 AI Pattern Adjustment (The Wiggles)</div>', unsafe_allow_html=True)
            st.info("🎯 This chart shows exactly how much the AI is adding or subtracting from the Excel baseline based on detected patterns.")
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Adjustment", marker_color="#00B0F0"
            ))
            fig_wig.update_layout(
                template="plotly_dark", 
                height=300, 
                title="Negative/Positive Patterns identified by AI",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_wig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- DATA TABLE & DOWNLOAD ---
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">📋 Forecasted Results Table</div>', unsafe_allow_html=True)
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "Predicted Calculated Forecast": predicted_calc_col,
                "Excel Calculated Forecast": excel_calc_col
            })
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='AI_Forecast')
            st.download_button(label="📥 Download Excel Result", data=output.getvalue(), file_name=f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
