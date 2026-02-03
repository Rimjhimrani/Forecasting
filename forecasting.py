import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Advanced Order Forecasting", layout="wide")

st.title("📊 ML & Statistical Order Forecasting")

# --- Helper Functions for ML ---
def create_features(df, label=None):
    """Creates time series features from datetime index."""
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # Create Lags (Previous 3 months)
    for i in range(1, 4):
        df[f'lag_{i}'] = df['order_qty'].shift(i)
    
    return df

def train_ml_model(model_type, train_df, horizon):
    """Trains ML model and predicts iteratively."""
    # Prepare training data (drop rows with NaN from lags)
    features = ['month', 'year', 'quarter', 'lag_1', 'lag_2', 'lag_3']
    df_clean = create_features(train_df).dropna()
    
    if len(df_clean) < 2:
        return [train_df['order_qty'].iloc[-1]] * horizon # Fallback to naive if data too small

    X = df_clean[features]
    y = df_clean['order_qty']

    # Initialize Model
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    elif model_type == "LightGBM":
        model = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbosity=-1)
    
    model.fit(X, y)

    # Iterative Prediction (using previous predictions as lags for future)
    preds = []
    last_known = df_clean.iloc[-1].copy()
    current_date = train_df['Date'].max()

    for i in range(horizon):
        current_date += pd.DateOffset(months=1)
        
        # Build feature row for prediction
        pred_row = pd.DataFrame([{
            'month': current_date.month,
            'year': current_date.year,
            'quarter': current_date.quarter,
            'lag_1': last_known['order_qty'] if i == 0 else preds[-1],
            'lag_2': df_clean['order_qty'].iloc[-1] if i == 0 else (last_known['order_qty'] if i == 1 else preds[-2]),
            'lag_3': df_clean['order_qty'].iloc[-2] if i == 0 else (df_clean['order_qty'].iloc[-1] if i == 1 else last_known['order_qty'] if i == 2 else preds[-3])
        }])
        
        res = model.predict(pred_row[features])[0]
        preds.append(max(0, res)) # Ensure no negative orders

    return preds

# --- Main Forecast Engine ---
def get_forecast(df, method, horizon, params):
    last_date = df['Date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    history = df['order_qty'].values.astype(float)
    
    if method in ["Random Forest", "XGBoost", "LightGBM"]:
        forecast_values = train_ml_model(method, df, horizon)
    
    elif method == "Historical Manager":
        forecast_values = [history[-1]] * horizon
        
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
        model = SimpleExpSmoothing(history, initialization_method="estimated").fit(smoothing_level=alpha, optimized=False)
        forecast_values = model.forecast(horizon)

    return pd.DataFrame({'Date': future_dates, 'order_qty': forecast_values, 'Type': 'Forecast'})

# --- UI Logic ---
uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    data.columns = data.columns.str.strip()
    
    required = ['Date', 'PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION', 'order_qty']
    column_mapping = {actual: req for req in required for actual in data.columns if req.lower() == actual.lower()}
    data = data.rename(columns=column_mapping)
    
    missing = [col for col in required if col not in data.columns]
    
    if missing:
        st.error(f"Missing: {missing}")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        data['order_qty'] = pd.to_numeric(data['order_qty'], errors='coerce').fillna(0)
        
        st.sidebar.header("Forecast Settings")
        selected_part = st.sidebar.selectbox("Select Part Number", data['PARTNO'].unique())
        method = st.sidebar.selectbox("Method", 
                                    ["Historical Manager", "Weightage Average", "Moving Average", 
                                     "Ramp up Average", "Exponentially", "Random Forest", "XGBoost", "LightGBM"])
        horizon = st.sidebar.slider("Months to Forecast", 1, 12, 6)
        
        # Method-specific settings
        params = {}
        if method in ["Moving Average", "Weightage Average"]:
            params['window'] = st.sidebar.number_input("Window", 1, 12, 3)
        elif method == "Ramp up Average":
            params['ramp_rate'] = st.sidebar.slider("Ramp %", 0, 50, 5) / 100
        elif method == "Exponentially":
            params['alpha'] = st.sidebar.slider("Alpha", 0.1, 1.0, 0.3)

        part_data = data[data['PARTNO'] == selected_part].sort_values('Date')
        
        # Execute Forecast
        try:
            forecast_df = get_forecast(part_data, method, horizon, params)
            
            # Combine info
            part_info = part_data.iloc[0].drop(['Date', 'order_qty']).to_dict()
            for col in ['PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION']:
                forecast_df[col] = part_info[col]

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=part_data['Date'], y=part_data['order_qty'], name='Actual Past', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['order_qty'], name=f'Forecast ({method})', line=dict(dash='dash', color='orange')))
            fig.update_layout(title=f"Forecasting for {selected_part}", xaxis_title="Timeline", yaxis_title="Quantity")
            st.plotly_chart(fig, use_container_width=True)

            # Table & Download
            st.subheader("Predicted Values")
            st.dataframe(forecast_df)
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast", csv, f"forecast_{selected_part}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.info("Ensure you have at least 4-6 months of historical data for ML models.")
