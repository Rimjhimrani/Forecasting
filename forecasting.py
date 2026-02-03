import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Order Forecasting System", layout="wide")

# --- CSS for better UI ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📦 Advanced Order Forecasting Dashboard")

# --- 1. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx'])

def clean_data(df):
    """Standardize column names and types"""
    df.columns = df.columns.str.strip()
    required = ['Date', 'PARTNO', 'PART DESCRIPTION', 'qty_veh_1', 'UOM', 'AGGREGATE', 'SUPPLY CONDITION', 'order_qty']
    
    # Map columns case-insensitively
    mapping = {col: req for req in required for col in df.columns if req.lower() == col.lower()}
    df = df.rename(columns=mapping)
    
    if all(col in df.columns for col in required):
        df['Date'] = pd.to_datetime(df['Date'])
        df['order_qty'] = pd.to_numeric(df['order_qty'], errors='coerce').fillna(0)
        return df, True
    return df, False

# --- 2. Forecasting Engines ---

def get_statistical_forecast(history, method, horizon, params):
    """Calculates the 5 specific business types"""
    if method == "Historical Manager":
        val = history[-1] if len(history) > 0 else 0
        return [val] * horizon
        
    elif method == "Moving Average":
        w = params.get('window', 3)
        val = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
        return [val] * horizon
        
    elif method == "Weightage Average":
        w = params.get('window', 3)
        if len(history) >= w:
            weights = np.arange(1, w + 1)
            val = np.dot(history[-w:], weights) / weights.sum()
        else:
            val = np.mean(history)
        return [val] * horizon
        
    elif method == "Ramp up Average":
        base_avg = np.mean(history)
        rate = params.get('ramp_rate', 0.05)
        return [base_avg * (1 + rate * i) for i in range(1, horizon + 1)]
        
    elif method == "Exponentially":
        alpha = params.get('alpha', 0.3)
        try:
            model = SimpleExpSmoothing(history).fit(smoothing_level=alpha, optimized=False)
            return model.forecast(horizon)
        except:
            return [history[-1]] * horizon

def get_ml_forecast(df, model_name, horizon):
    """ML Regressor Engine: XGBoost, LightGBM, Random Forest"""
    df = df.sort_values('Date').copy()
    
    # Feature Engineering
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    for i in range(1, 4): # Create 3 month lags
        df[f'lag_{i}'] = df['order_qty'].shift(i)
    
    train = df.dropna()
    if len(train) < 3: return [df['order_qty'].iloc[-1]] * horizon
    
    features = ['month', 'year', 'lag_1', 'lag_2', 'lag_3']
    X, y = train[features], train['order_qty']

    # Select Model
    if model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=100, verbosity=-1)
    else: # Random Forest
        model = RandomForestRegressor(n_estimators=100)

    model.fit(X, y)

    # Multi-step prediction
    preds = []
    last_row = train.iloc[-1]
    curr_lag1, curr_lag2, curr_lag3 = last_row['order_qty'], last_row['lag_1'], last_row['lag_2']
    curr_date = df['Date'].max()

    for i in range(horizon):
        curr_date += pd.DateOffset(months=1)
        pred_feat = pd.DataFrame([[curr_date.month, curr_date.year, curr_lag1, curr_lag2, curr_lag3]], columns=features)
        p = model.predict(pred_feat)[0]
        p = max(0, p)
        preds.append(p)
        curr_lag3, curr_lag2, curr_lag1 = curr_lag2, curr_lag1, p
        
    return preds

# --- 3. UI and Logic ---

if uploaded_file:
    raw_data, success = clean_data(pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file))
    
    if not success:
        st.error("Invalid file format. Ensure columns: Date, PARTNO, PART DESCRIPTION, qty_veh_1, UOM, AGGREGATE, SUPPLY CONDITION, order_qty")
    else:
        st.sidebar.header("Filter Selection")
        selected_part = st.sidebar.selectbox("Select PARTNO", raw_data['PARTNO'].unique())
        part_df = raw_data[raw_data['PARTNO'] == selected_part].sort_values('Date')
        
        st.sidebar.markdown("---")
        st.sidebar.header("1. Choose Forecast Type")
        stat_type = st.sidebar.selectbox("Business Method", ["Historical Manager", "Weightage Average", "Moving Average", "Ramp up Average", "Exponentially"])
        
        st.sidebar.header("2. Choose ML Model")
        ml_model_name = st.sidebar.selectbox("ML Regressor", ["XGBoost", "LightGBM", "Random Forest"])
        
        horizon = st.sidebar.slider("Forecast Months", 1, 12, 6)
        
        # Parameters for business logic
        params = {}
        if "Average" in stat_type: params['window'] = st.sidebar.number_input("Window size", 1, 12, 3)
        if "Ramp" in stat_type: params['ramp_rate'] = st.sidebar.slider("Ramp Rate %", 0, 50, 5)/100
        if "Exponentially" in stat_type: params['alpha'] = st.sidebar.slider("Alpha", 0.01, 1.0, 0.3)

        # Calculate both forecasts
        hist_vals = part_df['order_qty'].values
        stat_preds = get_statistical_forecast(hist_vals, stat_type, horizon, params)
        ml_preds = get_ml_forecast(part_df, ml_model_name, horizon)
        
        # Prepare future dates
        future_dates = [part_df['Date'].max() + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
        
        # --- 4. Visualization ---
        fig = go.Figure()
        # History
        fig.add_trace(go.Scatter(x=part_df['Date'], y=part_df['order_qty'], name="Past Actuals", line=dict(color="#2c3e50", width=3)))
        # Statistical Forecast
        fig.add_trace(go.Scatter(x=future_dates, y=stat_preds, name=f"Type: {stat_type}", line=dict(dash='dash', color='#e67e22')))
        # ML Forecast
        fig.add_trace(go.Scatter(x=future_dates, y=ml_preds, name=f"Model: {ml_model_name}", line=dict(dash='dot', color='#27ae60')))
        
        fig.update_layout(title=f"Comparison: {stat_type} vs {ml_model_name} for {selected_part}", 
                          hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. Download Table ---
        st.subheader("Forecast Comparison Data")
        
        # Building the final dataframe
        meta = part_df.iloc[0].drop(['Date', 'order_qty']).to_dict()
        export_df = pd.DataFrame({
            'Date': future_dates,
            'Statistical_Forecast': stat_preds,
            'ML_Forecast': ml_preds
        })
        for k, v in meta.items(): export_df[k] = v
        
        st.dataframe(export_df, use_container_width=True)
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast Result", csv, f"{selected_part}_forecast.csv", "text/csv")

else:
    st.info("Please upload a file to start.")
    st.image("https://via.placeholder.com/800x400.png?text=Upload+Data+to+View+Trend+Comparison", use_column_width=True)
