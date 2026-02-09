import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Machine Learning Imports
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title="ML Order Forecasting", layout="wide")

# --- UI Header ---
st.title("🚀 ML-Powered Order Forecasting (RF / XGB / LGBM)")
st.markdown("---")

# --- 1. Choose an Option ---
col1, col2 = st.columns(2)
with col1:
    main_option = st.radio("Choose Primary Option", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_option = None
if main_option == "Product Wise":
    with col2:
        sub_option = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- 2. Configuration Sidebar ---
with st.sidebar:
    st.header("Step 1: Configuration")
    interval = st.selectbox("Select Forecast Interval", ["Daily", "Weekly", "Monthly"])
    
    horizon_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
    horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_map.keys()))

    st.header("Step 2: ML Model Settings")
    technique = st.selectbox("Choose ML Model", ["Random Forest", "XGBoost", "LightGBM"])
    
    # Model Hyperparameters
    n_estimators = st.slider("Number of Trees (Estimators)", 50, 500, 100)
    max_depth = st.slider("Max Depth", 3, 20, 10)

# --- 3. Helper Functions for ML ---
def create_features(df):
    """Extracts features from the date index for ML models"""
    df = df.copy()
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    return df

# --- 4. Upload and Process Data ---
uploaded_file = st.file_uploader("Upload Data File (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load Raw Data
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw_df.columns = raw_df.columns.str.strip()

        # Handle Format 1 (Wide: Dates as Columns) or Format 2 (Long: Date column)
        if 'MODEL' in raw_df.columns and any('-' in col for col in raw_df.columns):
            # Wide format (Image 1)
            df_long = raw_df.melt(id_vars=['MODEL'], var_name='Date', value_name='order_qty')
            df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        else:
            # Long format (Image 2)
            df_long = raw_df.copy()
            df_long.rename(columns={'Part No': 'PARTNO', 'Qty/Veh': 'order_qty'}, inplace=True)
            df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')

        df_long = df_long.dropna(subset=['Date'])

        # Filter Logic
        if main_option == "Aggregate Wise":
            working_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            title_suffix = "Aggregate"
        elif sub_option == "Model Wise":
            selection = st.selectbox("Select Model", df_long['MODEL'].unique())
            working_df = df_long[df_long['MODEL'] == selection].copy()
            title_suffix = f"Model: {selection}"
        else:
            selection = st.selectbox("Select Part Number", df_long['PARTNO'].unique())
            working_df = df_long[df_long['PARTNO'] == selection].copy()
            title_suffix = f"Part: {selection}"

        # Resample
        resample_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        working_df = working_df.set_index('Date').resample(resample_map[interval])['order_qty'].sum().reset_index()
        
        if len(working_df) < 3:
            st.warning("Not enough historical data points to train a Machine Learning model. Please provide more dates.")
        else:
            # --- 5. Training Logic ---
            train_data = create_features(working_df)
            X = train_data[['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth']]
            y = train_data['order_qty']

            # Model Initialization
            if technique == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif technique == "XGBoost":
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1)
            else: # LightGBM
                model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1)

            model.fit(X, y)

            # --- 6. Forecasting Logic ---
            steps = horizon_map[horizon_label]
            if interval == "Weekly": steps = max(1, steps // 7)
            if interval == "Monthly": steps = max(1, steps // 30)

            last_date = working_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=resample_map[interval])[1:]
            
            # Create features for future dates
            future_df = pd.DataFrame({'Date': future_dates})
            future_features = create_features(future_df)[['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth']]
            
            # Prediction
            forecast_values = model.predict(future_features)
            forecast_values = np.maximum(forecast_values, 0) # Ensure no negative demand

            # --- 7. Visualization ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=working_df['Date'], y=working_df['order_qty'], name="Past Actuals", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, name=f"ML Forecast ({technique})", line=dict(color="#16a085", dash='dash')))
            
            fig.update_layout(title=f"Trend Analysis: {title_suffix} using {technique}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Results Table
            st.subheader("Forecasted Results")
            results_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "Predicted Qty": np.round(forecast_values, 2)})
            st.dataframe(results_df, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload your data file to begin the ML forecasting.")
