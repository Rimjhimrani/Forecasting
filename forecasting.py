import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from datetime import timedelta

# Set page configuration
st.set_page_config(page_title="Order Forecasting Dashboard", layout="wide")

st.title("📦 Order Forecasting Dashboard")
st.markdown("""
Upload your data file containing columns: **Date, PARTNO, PART DESCRIPTION, qty_veh_1, UOM, AGGREGATE, SUPPLY CONDITION, order_qty**.
""")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

def get_forecast(df, method, horizon, params):
    """Core forecasting logic applied per Part Number."""
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=30 * i) for i in range(1, horizon + 1)]
    history = df['order_qty'].values
    
    forecast_values = []
    
    if method == "Historical Manager":
        # Naive approach: Uses the last known value
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
        # Start from average and increase by a ramp percentage
        base_avg = np.mean(history)
        ramp_rate = params.get('ramp_rate', 0.05) # 5% increase per period
        forecast_values = [base_avg * (1 + ramp_rate * i) for i in range(1, horizon + 1)]
        
    elif method == "Exponentially":
        alpha = params.get('alpha', 0.3)
        model = SimpleExpSmoothing(history).fit(smoothing_level=alpha, optimized=False)
        forecast_values = model.forecast(horizon)

    return pd.DataFrame({
        'Date': future_dates,
        'order_qty': forecast_values,
        'Type': 'Forecast'
    })

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    # Pre-processing
    data['Date'] = pd.to_datetime(data['Date'])
    required_cols = ['Date', 'PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION', 'order_qty']
    
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing columns! Ensure file has: {', '.join(required_cols)}")
    else:
        st.success("Data Uploaded Successfully!")
        
        # --- 2. Sidebar Controls ---
        st.sidebar.header("Forecast Settings")
        selected_part = st.sidebar.selectbox("Select Part Number", data['PARTNO'].unique())
        method = st.sidebar.selectbox("Forecasting Method", 
                                    ["Historical Manager", "Weightage Average", "Moving Average", "Ramp up Average", "Exponentially"])
        horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
        
        # Dynamic parameters based on method
        params = {}
        if method in ["Moving Average", "Weightage Average"]:
            params['window'] = st.sidebar.number_input("Window Size", min_value=1, max_value=12, value=3)
        elif method == "Ramp up Average":
            params['ramp_rate'] = st.sidebar.slider("Ramp Growth Rate (%)", 0, 50, 5) / 100
        elif method == "Exponentially":
            params['alpha'] = st.sidebar.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)

        # Filter data for selected part
        part_data = data[data['PARTNO'] == selected_part].sort_values('Date')
        part_info = part_data.iloc[0].drop(['Date', 'order_qty']).to_dict()
        
        # Generate Forecast
        forecast_df = get_forecast(part_data, method, horizon, params)
        
        # Merge metadata back into forecast
        for col in ['PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION']:
            forecast_df[col] = part_info[col]
        
        # --- 3. Visualization ---
        fig = go.Figure()
        
        # Historical Data
        fig.add_trace(go.Scatter(x=part_data['Date'], y=part_data['order_qty'], 
                                 name='Past Actuals', mode='lines+markers', line=dict(color='blue')))
        
        # Forecast Data
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['order_qty'], 
                                 name=f'Future Forecast ({method})', mode='lines+markers', line=dict(dash='dash', color='red')))
        
        fig.update_layout(title=f"Trend Comparison for Part: {selected_part}",
                          xaxis_title="Date", yaxis_title="Order Quantity",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- 4. Data Display & Download ---
        st.subheader("Forecasted Data")
        st.write(forecast_df)
        
        # Download Button
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecasted Data as CSV",
            data=csv,
            file_name=f"forecast_{selected_part}_{method}.csv",
            mime='text/csv',
        )
