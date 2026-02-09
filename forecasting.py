import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# Set Page Config
st.set_page_config(page_title="Forecasting System", layout="centered")

# --- CUSTOM CSS FOR FLOWCHART UI ---
st.markdown("""
    <style>
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📦 Order Forecasting & Trend System")
st.info("Follow the flowchart steps below. The AI (XGBoost) will handle the rest automatically.")

# --- STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Select Category", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- STEP 2: SELECT FORECAST INTERVAL ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval</div>', unsafe_allow_html=True)
interval = st.selectbox("How frequently is data tracked?", 
                        ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"])

# --- STEP 3: SELECT FORECAST HORIZON ---
st.markdown('<div class="step-header">STEP 3: Select Forecast Horizon</div>', unsafe_allow_html=True)
# Restored "Day" as per flowchart requirement
horizon_label = st.selectbox("How far into the future should we predict?", 
                            ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"])

# --- STEP 4: UPLOAD THE DATA FILE ---
st.markdown('<div class="step-header">STEP 4: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV", type=['xlsx', 'csv'])

# --- INTERNAL XGBOOST ENGINE (Parameters Fixed for Non-Tech Users) ---
def generate_xgb_forecast(data, horizon_days, interval_type):
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
    
    # Pre-tuned XGBoost Parameters (Fixed)
    model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    # Determine number of steps to predict
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    freq = res_map[interval_type]
    
    # Calculate steps based on the frequency
    last_date = df['Date'].max()
    end_date = last_date + pd.Timedelta(days=horizon_days)
    
    # Ensure at least 1 step is predicted
    future_dates = pd.date_range(start=last_date, end=end_date, freq=freq)
    if len(future_dates) <= 1: # if horizon < interval
        future_dates = pd.date_range(start=last_date, periods=2, freq=freq)
    
    future_dates = future_dates[1:] # Drop the overlap with last historical date
    
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
        raw.columns = raw.columns.str.strip()

        # Handle formatting for both Image 1 (Wide) and Image 2 (Long)
        if 'MODEL' in raw.columns and any('-20' in str(c) for c in raw.columns):
            df_long = raw.melt(id_vars=['MODEL'], var_name='Date', value_name='order_qty')
            df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        else:
            df_long = raw.copy()
            df_long.rename(columns={'Part No': 'PARTNO', 'Qty/Veh': 'order_qty', 'Model': 'MODEL'}, inplace=True)
            if 'Time' in df_long.columns:
                df_long['Date'] = pd.to_datetime(df_long['Date'].astype(str) + ' ' + df_long['Time'].astype(str))
            else:
                df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')

        df_long = df_long.dropna(subset=['Date', 'order_qty'])

        # Filtering UI
        target_df = None
        item_label = "Aggregate Total"
        
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
        elif sub_choice == "Model Wise":
            selected_m = st.selectbox("Choose Model", df_long['MODEL'].unique())
            target_df = df_long[df_long['MODEL'] == selected_m].groupby('Date')['order_qty'].sum().reset_index()
            item_label = f"Model: {selected_m}"
        else:
            selected_p = st.selectbox("Choose Part Number", df_long['PARTNO'].unique())
            target_df = df_long[df_long['PARTNO'] == selected_p].groupby('Date')['order_qty'].sum().reset_index()
            item_label = f"Part: {selected_p}"

        # Consolidate data to the selected interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 5: GENERATE FORECAST AND TREND ANALYSIS ---
        st.markdown('<div class="step-header">STEP 5: Generate Forecast and Trend Analysis</div>', unsafe_allow_html=True)
        if st.button("🚀 Run AI Analysis", use_container_width=True):
            
            # Convert Horizon Label to total days
            horizon_days_map = {
                "Day": 1, "Week": 7, "Month": 30, "Quarter": 90, 
                "Year": 365, "3 years": 1095, "5 years": 1825
            }
            total_days = horizon_days_map[horizon_label]

            # Execute AI Forecast
            f_dates, f_values = generate_xgb_forecast(target_df, total_days, interval)

            # SHOW TREND ANALYSIS (LINE CHART)
            fig = go.Figure()
            # Historical line
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], 
                                     name="Past Performance", line=dict(color="#34495e", width=2)))
            # Forecast line
            fig.add_trace(go.Scatter(x=f_dates, y=f_values, 
                                     name="Future Forecast", line=dict(color="#e67e22", width=3, dash='dot')))
            
            fig.update_layout(title=f"Trend Analysis: {item_label}", 
                              xaxis_title="Date/Time", yaxis_title="Quantity",
                              template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # SHOW DATA TABLE
            st.subheader("📋 Predicted Quantities Table")
            date_fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
            res_df = pd.DataFrame({
                "Date/Time": f_dates.strftime(date_fmt), 
                "Predicted Quantity": np.round(f_values, 1)
            })
            st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong with the data file: {e}")
