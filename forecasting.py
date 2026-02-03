import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from datetime import timedelta

st.set_page_config(page_title="Order Forecasting Dashboard", layout="wide")

st.title("📦 Order Forecasting Dashboard")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

def get_forecast(df, method, horizon, params):
    last_date = df['Date'].max()
    # Create monthly future dates
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    history = df['order_qty'].values.astype(float)
    
    forecast_values = []
    
    if method == "Historical Manager":
        val = history[-1] if len(history) > 0 else 0
        forecast_values = [val] * horizon
        
    elif method == "Moving Average":
        window = params.get('window', 3)
        val = np.mean(history[-window:]) if len(history) >= window else np.mean(history)
        forecast_values = [val] * horizon
        
    elif method == "Weightage Average":
        window = params.get('window', 3)
        if len(history) >= window:
            weights = np.arange(1, window + 1)
            val = np.dot(history[-window:], weights) / weights.sum()
        else:
            val = np.mean(history)
        forecast_values = [val] * horizon
        
    elif method == "Ramp up Average":
        base_avg = np.mean(history)
        ramp_rate = params.get('ramp_rate', 0.05)
        forecast_values = [base_avg * (1 + ramp_rate * i) for i in range(1, horizon + 1)]
        
    elif method == "Exponentially":
        alpha = params.get('alpha', 0.3)
        try:
            model = SimpleExpSmoothing(history, initialization_method="estimated").fit(smoothing_level=alpha, optimized=False)
            forecast_values = model.forecast(horizon)
        except:
            forecast_values = [history[-1]] * horizon

    return pd.DataFrame({
        'Date': future_dates,
        'order_qty': forecast_values,
        'Type': 'Forecast'
    })

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    # --- Robust Column Cleaning ---
    # 1. Remove hidden spaces from column names
    data.columns = data.columns.str.strip()
    
    # 2. Define required columns
    required = ['Date', 'PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION', 'order_qty']
    
    # 3. Case-insensitive check: Try to map user columns to required columns
    column_mapping = {}
    for req in required:
        for actual in data.columns:
            if req.lower() == actual.lower():
                column_mapping[actual] = req
    
    # Rename columns to standard names
    data = data.rename(columns=column_mapping)
    
    # Check if all required columns now exist
    missing = [col for col in required if col not in data.columns]
    
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        st.info(f"The columns found in your file are: {list(data.columns)}")
    else:
        # Success - Pre-process data
        data['Date'] = pd.to_datetime(data['Date'])
        data['order_qty'] = pd.to_numeric(data['order_qty'], errors='coerce').fillna(0)
        
        # --- Sidebar ---
        st.sidebar.header("Forecast Settings")
        selected_part = st.sidebar.selectbox("Select Part Number", data['PARTNO'].unique())
        method = st.sidebar.selectbox("Forecasting Method", 
                                    ["Historical Manager", "Weightage Average", "Moving Average", "Ramp up Average", "Exponentially"])
        horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
        
        params = {}
        if method in ["Moving Average", "Weightage Average"]:
            params['window'] = st.sidebar.number_input("Window Size (Months)", min_value=1, max_value=12, value=3)
        elif method == "Ramp up Average":
            params['ramp_rate'] = st.sidebar.slider("Ramp Growth Rate (%)", 0, 50, 5) / 100
        elif method == "Exponentially":
            params['alpha'] = st.sidebar.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)

        # Filter and Forecast
        part_data = data[data['PARTNO'] == selected_part].sort_values('Date')
        part_info = part_data.iloc[0].drop(['Date', 'order_qty']).to_dict()
        
        forecast_df = get_forecast(part_data, method, horizon, params)
        
        # Apply metadata to forecast
        for col in ['PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION']:
            forecast_df[col] = part_info[col]
        
        # --- Visualization ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=part_data['Date'], y=part_data['order_qty'], name='Historical', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['order_qty'], name='Forecast', line=dict(dash='dash', color='red')))
        
        fig.update_layout(title=f"Forecast for {selected_part} ({method})", xaxis_title="Date", yaxis_title="Qty")
        st.plotly_chart(fig, use_container_width=True)

        # --- Download ---
        st.subheader("Forecast Results")
        st.dataframe(forecast_df)
        
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", csv, f"forecast_{selected_part}.csv", "text/csv")
