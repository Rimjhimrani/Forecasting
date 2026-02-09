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
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📦 Intelligent Order Forecasting")
st.info("System optimized for Wide Format Data. Powered by XGBoost AI.")

# --- STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- STEP 2: SELECT INTERVAL & HORIZON ---
st.markdown('<div class="step-header">STEP 2: Select Timeline</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    interval = st.selectbox("Forecast Interval", 
                            ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"])

with col_b:
    horizon_label = st.selectbox("Forecast Horizon", 
                                ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"])

# --- STEP 3: UPLOAD THE DATA FILE ---
st.markdown('<div class="step-header">STEP 3: Upload the Data File</div>', unsafe_allow_html=True)
st.caption("Upload the file where dates are across the top columns (Wide Format).")
uploaded_file = st.file_uploader("Choose Excel or CSV", type=['xlsx', 'csv'])

# --- INTERNAL AI ENGINE (XGBoost) ---
def run_ai_forecast(data, horizon_days, interval_type):
    df = data.copy()
    
    # Feature Engineering
    df['hour'] = df['Date'].dt.hour
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofweek'] = df['Date'].dt.dayofweek
    
    features = ['hour', 'day', 'month', 'year', 'dayofweek']
    X = df[features]
    y = df['order_qty']
    
    # Hidden Optimized XGBoost Parameters
    model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    # Setup Future Dates
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    freq = res_map[interval_type]
    
    last_date = df['Date'].max()
    end_date = last_date + pd.Timedelta(days=horizon_days)
    
    future_dates = pd.date_range(start=last_date, end=end_date, freq=freq)
    if len(future_dates) <= 1: 
        future_dates = pd.date_range(start=last_date, periods=2, freq=freq)
    
    future_dates = future_dates[1:] # Predictive steps
    
    f_df = pd.DataFrame({'Date': future_dates})
    f_df['hour'] = f_df['Date'].dt.hour
    f_df['day'] = f_df['Date'].dt.day
    f_df['month'] = f_df['Date'].dt.month
    f_df['year'] = f_df['Date'].dt.year
    f_df['dayofweek'] = f_df['Date'].dt.dayofweek
    
    preds = model.predict(f_df[features])
    return future_dates, np.maximum(preds, 0)

# --- DATA PROCESSING ---
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # 1. Wide to Long Transformation
        # First column is assumed to be the ID (MODEL or PART NO)
        id_col = raw.columns[0]
        df_long = raw.melt(id_vars=[id_col], var_name='Date', value_name='order_qty')
        
        # 2. Date Formatting
        df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        df_long = df_long.dropna(subset=['Date'])
        
        # 3. Handle Selection Filtering
        target_df = None
        item_label = "Total Aggregate"

        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
        else:
            options_list = df_long[id_col].unique()
            selected_item = st.selectbox(f"Select {id_col}", options_list)
            target_df = df_long[df_long[id_col] == selected_item].copy()
            item_label = f"{selected_item}"

        # 4. Consolidate into chosen interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 4: EXECUTION ---
        st.markdown('<div class="step-header">STEP 4: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            
            # Map Horizon to Days
            h_days = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
            
            # Run XGBoost
            f_dates, f_values = run_ai_forecast(target_df, h_days[horizon_label], interval)

            # CHARTING
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Actual Past Data", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=f_dates, y=f_values, name="XGBoost Prediction", line=dict(color="#28a745", dash='dot')))
            
            fig.update_layout(title=f"Trend Analysis for {item_label}", template="plotly_white", xaxis_title="Timeline")
            st.plotly_chart(fig, use_container_width=True)

            # DATA TABLE
            st.subheader("📋 Forecasted Results")
            fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
            res_df = pd.DataFrame({"Date": f_dates.strftime(fmt), "Predicted Qty": np.round(f_values, 1)})
            st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Format Error: Ensure the first column contains the Model names and other columns are dates. (Error: {e})")
