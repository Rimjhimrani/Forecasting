import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI Setup ---
st.set_page_config(page_title="AI Forecasting System", layout="centered")

st.markdown("""
    <style>
    .step-header {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin-bottom: 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Intelligent Order Forecasting")
st.info("Workflow following official flowchart. Powered by XGBoost AI.")

# --- STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- STEP 2: SELECT INTERVAL & HORIZON ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval & Horizon</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    interval = st.selectbox("Forecast Interval (Granularity)", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    horizon_label = st.selectbox("Forecast Horizon (Length)", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- STEP 3: CHOOSE FORECAST TECHNIQUE (XGBoost is the Engine for all) ---
st.markdown('<div class="step-header">STEP 3: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Method", [
    "Historical Average", 
    "Weightage Average", 
    "Moving Average", 
    "Ramp Up Evenly", 
    "Exponentially"
])

# Capture inputs based on your formulas
tech_inputs = {}
if technique == "Weightage Average":
    w_text = st.text_input("Manually enter weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
    tech_inputs['weights'] = [float(x.strip()) for x in w_text.split(',')]

elif technique == "Moving Average":
    tech_inputs['n'] = st.number_input("Lookback Window (n)", 2, 30, 3)

elif technique == "Ramp Up Evenly":
    tech_inputs['n'] = st.number_input("Ramp Window (n)", 2, 30, 3)
    tech_inputs['factor'] = st.number_input("Ramp Growth Factor", 1.0, 2.0, 1.05)

elif technique == "Exponentially":
    tech_inputs['alpha'] = st.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)

# --- STEP 4: UPLOAD DATA ---
st.markdown('<div class="step-header">STEP 4: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

# --- HIGH-SPEED XGBOOST ENGINE ---
def run_ai_engine(target_df, horizon_days, interval_type, tech, params):
    df = target_df.copy()
    
    # Feature Engineering
    df['h'], df['d'], df['m'], df['y'], df['dw'] = df['Date'].dt.hour, df['Date'].dt.day, df['Date'].dt.month, df['Date'].dt.year, df['Date'].dt.dayofweek
    features = ['h', 'd', 'm', 'y', 'dw']
    
    # ADJUSTING XGBOOST STRATEGY BASED ON FORMULAS
    sample_weights = np.ones(len(df))
    
    if tech == "Weightage Average":
        w = params['weights']
        # Apply weights to the most recent data points
        for i, weight in enumerate(reversed(w)):
            if (len(df)-1-i) >= 0: sample_weights[len(df)-1-i] = weight
            
    elif tech == "Moving Average":
        # Train AI only on the last 'n' points
        df = df.tail(params['n'])
        sample_weights = sample_weights[-len(df):]

    elif tech == "Exponentially":
        # Apply alpha-based weight decay (Recent data is more important)
        alpha = params['alpha']
        for i in range(len(sample_weights)):
            sample_weights[i] = (1 - alpha) ** (len(sample_weights) - 1 - i)

    # Train XGBoost
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(df[features], df['order_qty'], sample_weight=sample_weights)
    
    # Future dates logic
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=horizon_days), freq=res_map[interval_type])[1:]
    if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval_type])[1:]
    
    f_df = pd.DataFrame({'Date': future_dates})
    f_df['h'], f_df['d'], f_df['m'], f_df['y'], f_df['dw'] = f_df['Date'].dt.hour, f_df['Date'].dt.day, f_df['Date'].dt.month, f_df['Date'].dt.year, f_df['Date'].dt.dayofweek
    
    preds = model.predict(f_df[features])
    
    # Ramp Up Strategy logic
    if tech == "Ramp Up Evenly":
        factor = params['factor']
        preds = [p * (factor ** i) for i, p in enumerate(preds, 1)]
        
    return future_dates, np.maximum(preds, 0)

# --- EXECUTION ---
if uploaded_file:
    try:
        # 1. Process Wide Format
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # 2. Filter based on UI
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Aggregate"
        else:
            selected = st.selectbox(f"Select {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_label = str(selected)

        # 3. Resample
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 5: GENERATE ---
        st.markdown('<div class="step-header">STEP 5: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            with st.spinner('AI analyzing patterns...'):
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                
                f_dates, f_values = run_ai_engine(target_df, h_map[horizon_label], interval, technique, tech_inputs)

                # CHART
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Past data", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=f_dates, y=f_values, name=f"XGBoost Prediction ({technique})", line=dict(color="#e67e22", dash='dot')))
                fig.update_layout(title=f"Trend Analysis for {item_label}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # TABLE
                st.subheader("📋 Forecasted Table")
                fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                res_df = pd.DataFrame({"Date": f_dates.strftime(fmt), "Predicted Qty": np.round(f_values, 1)})
                st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
