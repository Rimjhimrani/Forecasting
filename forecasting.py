import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

st.set_page_config(page_title="XGBoost Order Forecasting", layout="wide")

# --- UI Header ---
st.title("🚀 XGBoost Machine Learning Forecasting")
st.markdown("---")

# --- 1. Navigation Options ---
col1, col2 = st.columns(2)
with col1:
    main_option = st.radio("Choose Primary Option", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_option = None
if main_option == "Product Wise":
    with col2:
        sub_option = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- 2. Configuration Sidebar ---
with st.sidebar:
    st.header("📈 Forecast Settings")
    interval = st.selectbox("Select Forecast Interval", ["Daily", "Weekly", "Monthly"])
    
    horizon_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
    horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_map.keys()))

    st.header("⚙️ XGBoost Parameters")
    n_estimators = st.slider("Boosting Rounds (Trees)", 50, 1000, 200)
    max_depth = st.slider("Tree Depth", 3, 15, 6)
    learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

# --- 3. Feature Engineering Function ---
def create_time_features(df):
    """Converts date into numerical features for XGBoost"""
    df = df.copy()
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    return df

# --- 4. Data Processing Logic ---
uploaded_file = st.file_uploader("Upload Data File (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load Raw Data
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw_df.columns = raw_df.columns.str.strip()

        # DETECT FORMAT AND CONVERT
        # Logic for Image 1 (Wide format: MODEL col + Date cols)
        if 'MODEL' in raw_df.columns and any('-20' in str(col) for col in raw_df.columns):
            df_long = raw_df.melt(id_vars=['MODEL'], var_name='Date', value_name='order_qty')
            df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        
        # Logic for Image 2 (Long format: Part No, Model, Date columns)
        else:
            df_long = raw_df.copy()
            # Standardize names from your second image
            col_rename = {'Part No': 'PARTNO', 'Qty/Veh': 'order_qty', 'Model': 'MODEL'}
            df_long.rename(columns=col_rename, inplace=True)
            df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')

        df_long = df_long.dropna(subset=['Date', 'order_qty'])

        # FILTERING BASED ON UI SELECTION
        if main_option == "Aggregate Wise":
            working_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            title_text = "Total Aggregate Demand"
        elif sub_option == "Model Wise":
            selection = st.selectbox("Select Model", df_long['MODEL'].unique())
            working_df = df_long[df_long['MODEL'] == selection].groupby('Date')['order_qty'].sum().reset_index()
            title_text = f"Model Demand: {selection}"
        else: # Part No Wise
            selection = st.selectbox("Select Part Number", df_long['PARTNO'].unique())
            working_df = df_long[df_long['PARTNO'] == selection].groupby('Date')['order_qty'].sum().reset_index()
            title_text = f"Part Demand: {selection}"

        # RESAMPLE DATA
        res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        working_df = working_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if len(working_df) < 5:
            st.error("Insufficient historical data for XGBoost (minimum 5 data points required).")
        else:
            # --- 5. MODEL TRAINING ---
            train_features = create_time_features(working_df)
            X = train_features[['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth']]
            y = train_features['order_qty']

            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(X, y)

            # --- 6. FORECASTING FUTURE ---
            # Calculate steps based on interval
            steps = horizon_map[horizon_label]
            if interval == "Weekly": steps = max(1, steps // 7)
            if interval == "Monthly": steps = max(1, steps // 30)

            last_date = working_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=res_map[interval])[1:]
            
            future_df = pd.DataFrame({'Date': future_dates})
            future_X = create_time_features(future_df)[['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth']]
            
            # Predict
            preds = model.predict(future_X)
            preds = np.maximum(preds, 0) # Remove negative forecasts

            # --- 7. VISUALIZATION ---
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(x=working_df['Date'], y=working_df['order_qty'], 
                                     name="Actual History", line=dict(color="#34495e", width=2)))
            # Forecast
            fig.add_trace(go.Scatter(x=future_dates, y=preds, 
                                     name="XGBoost Forecast", line=dict(color="#e74c3c", width=3, dash='dot')))
            
            fig.update_layout(title=f"Trend Analysis: {title_text}", 
                              xaxis_title="Date", yaxis_title="Quantity", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Display Data Table
            st.subheader("Forecasted Quantities")
            forecast_table = pd.DataFrame({
                "Forecast Date": future_dates.strftime('%d-%m-%Y'),
                "Predicted Quantity": np.round(preds, 2)
            })
            st.dataframe(forecast_table, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}. Please ensure column names match the uploaded images.")
else:
    st.info("Please upload your historical data file (Image 1 or Image 2 format) to generate an XGBoost forecast.")
