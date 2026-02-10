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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main { 
        background-color: #f8fafc;
    }
    
    .block-container {
        padding: 2rem 4rem;
        max-width: 1200px;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    
    .app-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #e2e8f0;
    }
    
    .step-badge {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        min-width: 40px;
        text-align: center;
    }
    
    .step-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Form Elements */
    label {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        background-color: white !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #3b82f6 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        gap: 1rem;
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    .stRadio label {
        background-color: white !important;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        background-color: white !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6 !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        background-color: white !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #3b82f6 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
        background-color: #f8fafc !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    
    /* Execute button */
    .big-button {
        margin: 3rem 0;
    }
    
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1.25rem !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 1.2rem !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Results section */
    .results-banner {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 3rem 0 2rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .results-banner h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Control panel */
    .control-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    .control-box-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #92400e;
        margin-bottom: 1rem;
    }
    
    /* Chart containers */
    .chart-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .chart-box h3 {
        color: #1e293b;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px !important;
        background-color: #dbeafe !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* Data table */
    .dataframe {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #ec4899 0%, #d946ef 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.875rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
        box-shadow: 0 2px 8px rgba(236, 72, 153, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4) !important;
    }
    
    /* Spacing */
    .row-spacing {
        margin-bottom: 1.5rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="app-header">
        <div class="app-title">📊 AI Precision Forecast</div>
        <div class="app-subtitle">Simple & Powerful Supply Chain Forecasting</div>
    </div>
""", unsafe_allow_html=True)

# --- STEP 1 ---
st.markdown("""
    <div class="section-header">
        <div class="step-badge">1</div>
        <div class="step-title">What do you want to forecast?</div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Select your forecasting scope", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Choose detail level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- STEP 2 ---
st.markdown("""
    <div class="section-header">
        <div class="step-badge">2</div>
        <div class="step-title">Set your timeline</div>
    </div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    interval = st.selectbox("How often do you want forecasts?", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_b:
    horizon_label = st.selectbox("How far into the future?", ["Day", "Week", "Month", "Quarter", "Year"], index=2)

# --- STEP 3 ---
st.markdown("""
    <div class="section-header">
        <div class="step-badge">3</div>
        <div class="step-title">Choose your calculation method</div>
    </div>
""", unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    technique = st.selectbox("Select baseline strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
with col_d:
    if technique == "Weightage Average":
        w_in = st.text_input("Enter weights (comma separated)", "0.2, 0.3, 0.5")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Number of periods to average", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth multiplier", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing factor", 0.01, 1.0, 0.3)

# --- STEP 4 ---
st.markdown("""
    <div class="section-header">
        <div class="step-badge">4</div>
        <div class="step-title">Upload your data</div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop your Excel or CSV file here (dates should be column headers)", type=['xlsx', 'csv'])

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
            selected = st.selectbox(f"Select the item you want to forecast", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast"):
            st.session_state.run_analysis = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('run_analysis', False):
            
            # --- RESULTS BANNER ---
            st.markdown("""
                <div class="results-banner">
                    <h2>✨ Your Forecast is Ready!</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # --- CONTROL PANEL ---
            st.markdown('<div class="control-box">', unsafe_allow_html=True)
            st.markdown('<div class="control-box-title">💡 Want to change the forecast period? Adjust it here:</div>', unsafe_allow_html=True)
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Number of periods", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Period type", ["Days", "Weeks", "Months", "Original Selection"])
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
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown(f'<h3>📈 Forecast Trend: {item_name}</h3>', unsafe_allow_html=True)
            
            fig = go.Figure()

            # TRADED
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Actual History",
                mode='lines+markers', line=dict(color="#3b82f6", width=3, shape='spline'),
                marker=dict(size=7, color="white", line=dict(color="#3b82f6", width=2))
            ))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            # EXCEL BASELINE
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Basic Forecast",
                mode='lines+markers', line=dict(color="#94a3b8", width=2, dash='dot', shape='spline'),
                marker=dict(size=5, color="#94a3b8")
            ))

            # AI PREDICTION
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Smart Forecast",
                mode='lines+markers', line=dict(color="#10b981", width=3, dash='dash', shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#10b981", width=2))
            ))

            fig.add_vline(x=last_date, line_width=2, line_dash="dash", line_color="#e2e8f0", annotation_text="Today")
            
            fig.update_layout(
                template="plotly_white", 
                hovermode="x unified", 
                height=500, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- AI WIGGLE CHART ---
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown('<h3>🎯 AI Pattern Detection</h3>', unsafe_allow_html=True)
            st.info("This shows how the AI is adjusting the forecast based on patterns it detected in your data")
            
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Adjustment", 
                marker_color=['#ef4444' if x < 0 else '#10b981' for x in ai_residuals]
            ))
            fig_wig.update_layout(
                template="plotly_white", 
                height=300,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_wig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- DATA TABLE & DOWNLOAD ---
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown('<h3>📋 Detailed Forecast Data</h3>', unsafe_allow_html=True)
            
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "AI Smart Forecast": predicted_calc_col,
                "Basic Forecast": excel_calc_col
            })
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='Forecast')
            
            st.download_button(
                label="📥 Download Forecast as Excel", 
                data=output.getvalue(), 
                file_name=f"Forecast_{item_name}.xlsx",
                mime="application/vnd.ms-excel"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")
        st.info("Please make sure your file has dates as column headers and data is in the correct format.")
