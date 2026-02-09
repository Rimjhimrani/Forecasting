import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI Setup ---
st.set_page_config(page_title="Forecasting System", layout="centered")

st.markdown("""
    <style>
    .step-header {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        margin-bottom: 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📦 Intelligent Order Forecasting System")

# --- FLOWCHART STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Option", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Product Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- FLOWCHART STEP 2: SELECT FORECAST INTERVAL ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval</div>', unsafe_allow_html=True)
interval = st.selectbox("Frequency", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)

# --- FLOWCHART STEP 3: SELECT FORECAST HORIZON ---
st.markdown('<div class="step-header">STEP 3: Select Forecast Horizon</div>', unsafe_allow_html=True)
horizon_label = st.selectbox("Timeline", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: CHOOSE FORECAST TECHNIQUES ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Technique (Powered by XGBoost AI)", [
    "Historical Average", 
    "Weightage Average", 
    "Moving Average", 
    "Ramp Up Evenly", 
    "Exponentially"
])

# Capture flowchart-specific sub-inputs
tech_params = {}
if technique == "Weightage Average":
    w_mode = st.radio("Weight Calculation", ["Automated (Even Weights)", "Manual entering of weights"], horizontal=True)
    if w_mode == "Manual entering of weights":
        w_input = st.text_input("Enter weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
        tech_params['weights'] = [float(x.strip()) for x in w_input.split(',')]
    else:
        tech_params['window'] = st.number_input("Lookback window for automated weights", 2, 12, 3)

elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Select n (Lookback Window)", 2, 30, 7)

elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Manually entering of Interval Ramp up Factor", 1.0, 3.0, 1.05, 0.01)

# --- FLOWCHART STEP 5: UPLOAD THE DATA FILE ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

# --- CORE XGBOOST SCM ENGINE ---
def run_scm_engine(data, horizon_days, interval_type, tech, params):
    df = data.copy()
    
    # 1. Feature Engineering (The signals for XGBoost)
    df['f_hist_avg'] = df['order_qty'].expanding().mean().shift(1)
    df['f_ma_7'] = df['order_qty'].rolling(window=7, min_periods=1).mean().shift(1)
    df['f_exp_smooth'] = df['order_qty'].ewm(alpha=0.3).mean().shift(1)
    
    # Time signals
    df['h'], df['d'], df['m'], df['y'], df['dw'] = df['Date'].dt.hour, df['Date'].dt.day, df['Date'].dt.month, df['Date'].dt.year, df['Date'].dt.dayofweek
    
    features = ['f_hist_avg', 'f_ma_7', 'f_exp_smooth', 'h', 'd', 'm', 'y', 'dw']
    X = df[features].fillna(method='bfill').fillna(0)
    y = df['order_qty']

    # Sample Weighting for Weightage Average Technique
    sw = np.ones(len(df))
    if tech == "Weightage Average" and 'weights' in params:
        weights = params['weights']
        for i, w in enumerate(reversed(weights)):
            if (len(sw)-1-i) >= 0: sw[len(sw)-1-i] = w

    # Train XGBoost
    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    model.fit(X, y, sample_weight=sw)
    
    # Setup Future dates
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=horizon_days), freq=res_map[interval_type])[1:]
    if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval_type])[1:]
    
    # Prepare future features (use last known signals)
    f_df = pd.DataFrame({'Date': future_dates})
    f_df['h'], f_df['d'], f_df['m'], f_df['y'], f_df['dw'] = f_df['Date'].dt.hour, f_df['Date'].dt.day, f_df['Date'].dt.month, f_df['Date'].dt.year, f_df['Date'].dt.dayofweek
    
    # Fill signal columns with last historical values
    for col in ['f_hist_avg', 'f_ma_7', 'f_exp_smooth']:
        f_df[col] = X[col].iloc[-1]
    
    preds = model.predict(f_df[features])

    # Technique Specific Adjustment: Ramp Up Evenly
    if tech == "Ramp Up Evenly":
        factor = params['ramp_factor']
        preds = [p * (factor ** i) for i, p in enumerate(preds, 1)]

    return future_dates, np.maximum(preds, 0)

# --- EXECUTION ---
if uploaded_file:
    try:
        # Load and Transform Wide format
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Aggregate Demand"
        else:
            options = df_long[id_col].unique()
            selected = st.selectbox(f"Select {id_col}", options)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_label = str(selected)

        # Consolidate
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- FLOWCHART STEP 6: GENERATE ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast and Trend Analysis", use_container_width=True):
            if len(target_df) < 3:
                st.error("Insufficient historical data points.")
            else:
                with st.spinner('Analyzing patterns with XGBoost...'):
                    h_days = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                    f_dates, f_values = run_scm_engine(target_df, h_days[horizon_label], interval, technique, tech_params)

                    # Show Trend Analysis (Line Chart)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Past data", line=dict(color="#2c3e50")))
                    fig.add_trace(go.Scatter(x=f_dates, y=f_values, name=f"AI Forecast ({technique})", line=dict(color="#28a745", dash='dot')))
                    fig.update_layout(title=f"Trend Analysis for {item_label}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    # Show Results Table
                    st.subheader("📋 Predicted Quantities")
                    date_fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                    res_df = pd.DataFrame({"Date": f_dates.strftime(date_fmt), "Predicted Qty": np.round(f_values, 1)})
                    st.dataframe(res_df, use_container_width=True)
                    
                    st.success("Analysis Complete (END of Process)")

    except Exception as e:
        st.error(f"Error: {e}")
