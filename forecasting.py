import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Order Forecasting", layout="wide")

st.markdown("""
<style>
    /* Main background */
    .main { 
        background-color: #f5f5f5;
        padding: 20px;
    }
    
    /* Header section with robot */
    .header-section {
        background: linear-gradient(135deg, #5b9bd5 0%, #4a7fb8 100%);
        padding: 50px;
        border-radius: 20px;
        margin-bottom: 30px;
        color: white;
        position: relative;
    }
    
    .header-title {
        display: flex;
        align-items: center;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .header-icon {
        font-size: 1.2em;
        margin-right: 20px;
    }
    
    .header-subtitle {
        font-size: 1.2em;
        opacity: 0.95;
        margin-left: 80px;
    }
    
    /* Info banner */
    .info-banner {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 5px solid #28a745;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .info-icon {
        font-size: 2em;
        margin-right: 15px;
    }
    
    .info-text {
        color: #155724;
        font-size: 1em;
    }
    
    /* Step cards */
    .step-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 2px solid #e9ecef;
    }
    
    .step-badge {
        background: linear-gradient(135deg, #5a7a8f 0%, #4a6a7f 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 1.2em;
        font-weight: 700;
        margin-right: 15px;
        box-shadow: 0 2px 8px rgba(90, 122, 143, 0.3);
    }
    
    .step-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #2c5f7f;
    }
    
    /* Two column layout for steps */
    .two-col-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin-top: 20px;
    }
    
    /* Upload card */
    .upload-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .upload-area {
        background: white;
        border: 3px dashed #90caf9;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    
    .upload-icon {
        font-size: 4em;
        color: #64b5f6;
        margin-bottom: 15px;
    }
    
    /* Execute card */
    .execute-card {
        background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .execute-icon {
        font-size: 4em;
        margin-bottom: 15px;
    }
    
    /* Buttons */
    .pill-button {
        background: linear-gradient(135deg, #5a9c8e 0%, #4a8c7e 100%);
        color: white;
        padding: 12px 30px;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        font-size: 1em;
    }
    
    .outline-button {
        background: white;
        color: #6c757d;
        padding: 12px 30px;
        border-radius: 25px;
        border: 2px solid #dee2e6;
        font-weight: 600;
        font-size: 1em;
    }
    
    /* Execute button */
    .execute-btn > button {
        width: 100% !important;
        background: linear-gradient(135deg, #5a9c8e 0%, #4a8c7e 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 18px 40px !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 1.2em !important;
        box-shadow: 0 4px 12px rgba(90, 156, 142, 0.3) !important;
    }
    
    /* Section labels */
    .section-label {
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    /* Tip box */
    .tip-box {
        background: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        display: flex;
        align-items: center;
    }
    
    .tip-icon {
        font-size: 1.5em;
        margin-right: 10px;
    }
    
    /* Bottom info */
    .bottom-info {
        background: #e7f3ff;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        display: flex;
        align-items: center;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
    }
    
    /* Input styling */
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="header-section">
    <div class="header-title">
        <span class="header-icon">📦</span>
        AI Order Forecasting
    </div>
    <div class="header-subtitle">Smart predictions to optimize your supply chain.</div>
</div>
""", unsafe_allow_html=True)

# --- INFO BANNER ---
st.markdown("""
<div class="info-banner">
    <span class="info-icon">💡</span>
    <span class="info-text">Harness AI & baseline formulas to achieve pinpoint forecasting accuracy.</span>
</div>
""", unsafe_allow_html=True)

# --- STEP 1-3: CONFIGURE ---
st.markdown("""
<div class="step-card">
    <div class="step-header">
        <div class="step-badge">1-3</div>
        <div class="step-title">Configure</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-label">AGGREGATION</p>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    main_choice = st.radio("agg", ["Aggregate Wise", "Product Wise"], horizontal=True, label_visibility="collapsed")
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("spec", ["Model Wise", "Part No Wise"], horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<p class="section-label">Interval</p>', unsafe_allow_html=True)
    interval = st.selectbox("int", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1, label_visibility="collapsed")
with col_b:
    st.markdown('<p class="section-label">Horizon</p>', unsafe_allow_html=True)
    horizon_label = st.selectbox("hor", ["Day", "Week", "Month", "Quarter", "Year"], index=2, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<p class="section-label">Techniques</p>', unsafe_allow_html=True)
technique = st.selectbox("tech", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"], label_visibility="collapsed")

tech_params = {}
if technique == "Weightage Average":
    w_in = st.text_input("Weights (comma separated)", "0.2, 0.3, 0.5")
    try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
    except: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
elif technique == "Moving Average":
    st.markdown('<p class="section-label">Window Size (n)</p>', unsafe_allow_html=True)
    tech_params['n'] = st.slider("n", 2, 30, 7, label_visibility="collapsed")
elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Alpha", 0.01, 1.0, 0.3)

st.markdown("""
<div class="tip-box">
    <span class="tip-icon">💡</span>
    <span>Try Ramp Up Evenly or Exponentially for advanced adjustments.</span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- TWO COLUMN LAYOUT FOR UPLOAD AND EXECUTE ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="upload-card">
        <div class="step-header">
            <div class="step-badge">2</div>
            <div class="step-title">Step 4-5: Upload Data</div>
        </div>
        <div class="upload-area">
            <div class="upload-icon">☁️</div>
            <h3 style="margin: 10px 0;">Upload Excel / CSV file</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("file", type=['xlsx', 'csv'], label_visibility="collapsed")
    
    st.markdown("""
        <p style="text-align: center; color: #6c757d; margin-top: 15px;">
            Wide-format data. Compatible with xlsx, xls, or csv
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="execute-card">
        <div class="step-header">
            <div class="step-badge" style="background: linear-gradient(135deg, #d4a944 0%, #c49934 100%);">6</div>
            <div class="step-title">Step 6:</div>
        </div>
        <div class="execute-icon">🚀</div>
        <h3 style="margin: 20px 0;">Generate Forecast</h3>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
    exec_button = st.button("🚀 Generate Forecast")
    if exec_button:
        st.session_state.run_analysis = True
    st.markdown('</div>', unsafe_allow_html=True)
    
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
            selected = st.selectbox(f"Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.session_state.get('run_analysis', False):
            st.divider()
            
            # --- DYNAMIC HORIZON CONTROL ---
            st.markdown("""
            <div class="step-card">
                <h3 style="color: #2c5f7f; margin-bottom: 15px;">🔄 Adjust Forecast Horizon</h3>
                <p style="color: #6c757d;">Change the forecast period instantly</p>
            """, unsafe_allow_html=True)
            
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Enter Quantity", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Select Unit", ["Days", "Weeks", "Months", "Original Selection"])
            
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
            st.markdown('<h2 style="color: #2c5f7f; margin: 30px 0 20px 0;">📈 Predictive Trend Analysis: ' + item_name + '</h2>', unsafe_allow_html=True)
            fig = go.Figure()

            # TRADED
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Traded",
                mode='lines+markers', line=dict(color="#5b9bd5", width=3, shape='spline'),
                marker=dict(size=7, color="white", line=dict(color="#5b9bd5", width=2))
            ))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            # EXCEL BASELINE
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Excel Calculated Forecast",
                mode='lines+markers', line=dict(color="#a6a6a6", width=2, dash='dot', shape='spline'),
                marker=dict(size=5, color="#a6a6a6")
            ))

            # AI PREDICTION
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Predicted Forecast",
                mode='lines+markers', line=dict(color="#ffa500", width=3, dash='dash', shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#ffa500", width=2))
            ))

            fig.add_vline(x=last_date, line_width=2, line_color="#cccccc", line_dash="dash")
            
            fig.update_layout(
                template="plotly_white", 
                hovermode="x unified", 
                height=500,
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- AI WIGGLE CHART ---
            st.markdown('<h2 style="color: #2c5f7f; margin: 30px 0 20px 0;">📉 AI Pattern Adjustment (The Wiggles)</h2>', unsafe_allow_html=True)
            st.info("📊 This chart shows exactly how much the AI is adding or subtracting from the Excel baseline based on detected patterns.")
            
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Adjustment", 
                marker_color=['#5a9c8e' if x >= 0 else '#e74c3c' for x in ai_residuals]
            ))
            fig_wig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- DATA TABLE & DOWNLOAD ---
            st.markdown('<h2 style="color: #2c5f7f; margin: 30px 0 20px 0;">📋 Forecasted Results Table</h2>', unsafe_allow_html=True)
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

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# --- BOTTOM INFO ---
st.markdown("""
<div class="bottom-info">
    <span style="font-size: 1.5em; margin-right: 10px;">ℹ️</span>
    <span>Try Ramp Up Evenly or Exponentially for advanced adjustments.</span>
</div>
""", unsafe_allow_html=True)
