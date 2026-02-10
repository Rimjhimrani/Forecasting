import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide")

st.markdown("""
<style>
    /* Main background with gradient */
    .main { 
        background: linear-gradient(180deg, #f0f4ff 0%, #e8f5f7 100%);
        padding-bottom: 50px;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(74, 144, 226, 0.3);
        color: white;
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 10px;
        color: white;
    }
    
    .hero-subtitle {
        font-size: 1.2em;
        opacity: 0.95;
        font-weight: 400;
    }
    
    /* Info banner */
    .info-banner {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 20px 25px;
        border-radius: 15px;
        margin-bottom: 30px;
        border-left: 5px solid #4caf50;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    .info-icon {
        font-size: 2em;
        margin-right: 15px;
    }
    
    .info-text {
        color: #2e7d32;
        font-size: 1.05em;
        font-weight: 500;
    }
    
    /* Step container */
    .step-container {
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
        border: 2px solid #e3f2fd;
    }
    
    /* Step header */
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 25px;
    }
    
    .step-badge {
        background: linear-gradient(135deg, #5c7cba 0%, #4a5f8f 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 24px;
        font-weight: bold;
        margin-right: 15px;
        box-shadow: 0 4px 12px rgba(92, 124, 186, 0.4);
    }
    
    .step-title {
        font-size: 1.6em;
        font-weight: 700;
        color: #2c3e50;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 3px dashed #64b5f6;
        margin: 20px 0;
    }
    
    .upload-icon {
        font-size: 4em;
        margin-bottom: 15px;
        opacity: 0.7;
    }
    
    /* Execute section */
    .execute-section {
        background: linear-gradient(135deg, #fff9e6 0%, #ffe6b3 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 6px 25px rgba(255, 193, 7, 0.2);
    }
    
    .execute-icon {
        font-size: 4em;
        margin-bottom: 15px;
    }
    
    /* Execute button */
    .execute-btn > button {
        width: 100% !important;
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%) !important;
        color: white !important;
        font-weight: bold !important;
        padding: 20px 40px !important;
        border-radius: 15px !important;
        border: none !important;
        font-size: 1.4em !important;
        box-shadow: 0 6px 20px rgba(67, 160, 71, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .execute-btn > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(67, 160, 71, 0.6) !important;
    }
    
    /* Dynamic control section */
    .dynamic-section {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 30px;
        border-radius: 20px;
        margin: 25px 0;
        border: 2px solid #ffb74d;
    }
    
    .dynamic-title {
        font-size: 1.3em;
        font-weight: 700;
        color: #e65100;
        margin-bottom: 15px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        border: none !important;
        font-size: 1.1em !important;
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30, 136, 229, 0.5) !important;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #1e3a8a;
        margin: 30px 0 20px 0;
        display: flex;
        align-items: center;
    }
    
    /* Info alerts */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #4a90e2;
        background-color: #e3f2fd;
    }
    
    /* Input styling */
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #f5f5f5;
        padding: 10px 15px;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        margin: 40px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4a90e2, transparent);
    }
</style>
""", unsafe_allow_html=True)


# Hero Section
st.markdown("""
<div class="hero-section">
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
        <span style="font-size: 3em; margin-right: 20px;">📦</span>
        <h1 class="hero-title">AI Order Forecasting</h1>
    </div>
    <p class="hero-subtitle">Smart predictions to optimize your supply chain.</p>
</div>
""", unsafe_allow_html=True)

# Info Banner
st.markdown("""
<div class="info-banner">
    <span class="info-icon">💡</span>
    <span class="info-text">Harness AI & baseline formulas to achieve pinpoint forecasting accuracy.</span>
</div>
""", unsafe_allow_html=True)

# --- 2. STEP 1-3: CONFIGURATION ---
st.markdown("""
<div class="step-container">
    <div class="step-header">
        <div class="step-badge">1-3</div>
        <div class="step-title">Configure</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("**AGGREGATION**")
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Primary Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True, label_visibility="collapsed")
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Specific Level", ["Model Wise", "Part No Wise"], horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Interval**")
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1, label_visibility="collapsed")
with col_b:
    st.markdown("**Horizon**")
    horizon_label = st.selectbox("Default Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("**Techniques**")
col_c, col_d = st.columns([2, 1])
with col_c:
    technique = st.selectbox("Excel Strategy (Baseline)", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"], label_visibility="collapsed")

tech_params = {}
with col_d:
    if technique == "Weightage Average":
        w_in = st.text_input("Weights", "0.2, 0.3, 0.5", label_visibility="collapsed")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Window Size (n)", 2, 30, 7, label_visibility="collapsed")
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05, label_visibility="collapsed")
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Alpha", 0.01, 1.0, 0.3, label_visibility="collapsed")

if technique in ["Ramp Up Evenly", "Exponentially"]:
    st.markdown("💡 <small>Try Ramp Up Evenly or Exponentially for advanced adjustments.</small>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- 3. STEP 4-5: UPLOAD DATA ---
st.markdown("""
<div class="step-container">
    <div class="step-header">
        <div class="step-badge">2</div>
        <div class="step-title">Step 4-5: Upload Data</div>
    </div>
    <div class="upload-section">
        <div class="upload-icon">☁️</div>
        <h3>Upload Excel / CSV file</h3>
        <p style="color: #666; margin-top: 10px;">Wide-format data. Compatible with xlsx, xls, or csv</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'], label_visibility="collapsed")

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

# --- 7. EXECUTION ---
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
            selected = st.selectbox(f"Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # Execute Section
        st.markdown("""
        <div class="execute-section">
            <div class="execute-icon">🚀</div>
            <h2 style="margin: 0; color: #2c3e50;">Step 6: Generate Forecast</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast"):
            st.session_state.run_analysis = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('run_analysis', False):
            st.divider()
            
            # --- DYNAMIC HORIZON CONTROL ---
            st.markdown("""
            <div class="dynamic-section">
                <div class="dynamic-title">🔄 Adjust Forecast Horizon</div>
                <p style="color: #666; margin-bottom: 15px;">Change the forecast period instantly</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Enter Quantity", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Select Unit", ["Days", "Weeks", "Months", "Original Selection"])

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

            # --- 8. TREND GRAPH (Premium Curvy Style) ---
            st.markdown(f'<div class="section-title">📈 Predictive Trend Analysis: {item_name}</div>', unsafe_allow_html=True)
            fig = go.Figure()

            # TRADED
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Traded",
                mode='lines+markers', line=dict(color="#667eea", width=3, shape='spline'),
                marker=dict(size=7, color="white", line=dict(color="#667eea", width=2))
            ))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            # EXCEL BASELINE (Baseline)
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Excel Calculated Forecast",
                mode='lines+markers', line=dict(color="#a0aec0", width=2, dash='dot', shape='spline'),
                marker=dict(size=5, color="#a0aec0")
            ))

            # AI PREDICTION (Final)
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Predicted Forecast",
                mode='lines+markers', line=dict(color="#f59e0b", width=3, dash='dash', shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#f59e0b", width=2))
            ))

            fig.add_vline(x=last_date, line_width=2, line_color="#cbd5e0", line_dash="dash")
            fig.add_annotation(x=target_df['Date'].iloc[int(len(target_df)*0.8)], y=target_df['qty'].max()*1.1, text="🛍️", showarrow=False, bgcolor="rgba(102,126,234,0.15)", bordercolor="#667eea", borderwidth=2, borderpad=8)
            fig.add_annotation(x=future_dates[int(len(future_dates)*0.5)] if len(future_dates)>0 else last_date, y=max(predicted_calc_col)*1.1 if len(predicted_calc_col)>0 else last_qty, text="📢", showarrow=False, bgcolor="rgba(245,158,11,0.15)", bordercolor="#f59e0b", borderwidth=2, borderpad=8)

            fig.update_layout(
                template="plotly_white", 
                hovermode="x unified", 
                height=550,
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0",
                    borderwidth=1
                ),
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 9. AI WIGGLE CHART (The Seasonal Patterns) ---
            st.markdown('<div class="section-title">📉 AI Pattern Adjustment (The Wiggles)</div>', unsafe_allow_html=True)
            st.info("📊 This chart shows exactly how much the AI is adding or subtracting from the Excel baseline based on detected patterns.")
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Adjustment", 
                marker_color=['#10b981' if x >= 0 else '#ef4444' for x in ai_residuals],
                marker_line_color='white',
                marker_line_width=1.5
            ))
            fig_wig.update_layout(
                template="plotly_white", 
                height=350, 
                title="Negative/Positive Patterns identified by AI",
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- 10. DATA TABLE & DOWNLOAD ---
            st.markdown('<div class="section-title">📋 Forecasted Results Table</div>', unsafe_allow_html=True)
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "Predicted Calculated Forecast": predicted_calc_col,
                "Excel Calculated Forecast": excel_calc_col
            })
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='AI_Forecast')
            st.download_button(label="📥 Download Excel Result (3 Columns)", data=output.getvalue(), file_name=f"Forecast_{item_name}.xlsx")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
