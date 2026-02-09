import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI Setup ---
st.set_page_config(page_title="AI SCM Forecasting", layout="centered")

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

st.title("🚀 AI-Powered Order Forecasting")
st.info("Classical formulas transformed into XGBoost features for professional accuracy.")

# --- FLOWCHART STEPS 1-4 ---
st.markdown('<div class="step-header">STEPS 1-4: Configuration</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    main_choice = st.radio("Selection", ["Aggregate Wise", "Product Wise"], horizontal=True)
    interval = st.selectbox("Interval (Spacing)", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    
with col2:
    sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True) if main_choice == "Product Wise" else None
    horizon_label = st.selectbox("Horizon (Length)", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# Technique Selection (Becomes the 'Primary Signal' for XGBoost)
technique = st.selectbox("Forecast Technique (AI Strategy)", 
                         ["Historical Average", "Moving Average", "Weightage Average", "Ramp Up Evenly", "Exponentially"])

# --- FORMULA CALCULATORS (Your provided logic) ---
def get_weighted_avg(series, weights):
    n = len(weights)
    if len(series) < n: return np.mean(series)
    return np.dot(series[-n:], weights) / np.sum(weights)

def get_ramp_up(series):
    n = len(series)
    weights = np.arange(1, n + 1)
    return np.dot(series, weights) / weights.sum()

# --- XGBOOST FEATURE ENGINE ---
def build_scm_features(df, tech, h_days):
    """Turns classical formulas into XGBoost input features"""
    data = df.copy()
    
    # 1. Historical Average Feature
    data['feat_hist_avg'] = data['order_qty'].expanding().mean().shift(1)
    
    # 2. Moving Average Features (Multiple windows for robustness)
    for n in [3, 7, 14]:
        data[f'feat_ma_{n}'] = data['order_qty'].rolling(window=n).mean().shift(1)
        
    # 3. Weighted Average (Standard 3-period)
    data['feat_weighted'] = data['order_qty'].rolling(window=3).apply(lambda x: get_weighted_avg(x, [0.2, 0.3, 0.5]), raw=True).shift(1)
    
    # 4. Ramp Up Feature (Linear momentum)
    data['feat_ramp'] = data['order_qty'].rolling(window=7).apply(get_ramp_up, raw=True).shift(1)
    
    # 5. Exponential Smoothing Feature (Alpha 0.3)
    data['feat_exp'] = data['order_qty'].ewm(alpha=0.3).mean().shift(1)
    
    # 6. Time Features
    data['month'] = data['Date'].dt.month
    data['dayofweek'] = data['Date'].dt.dayofweek
    
    return data.dropna()

# --- STEP 5: UPLOAD & PROCESS ---
st.markdown('<div class="step-header">STEP 5: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        
        # Melt Wide to Long
        valid_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=valid_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_name = "Aggregate"
        else:
            selected = st.selectbox(f"Select {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Resample
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: EXECUTION ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Run AI Analysis", use_container_width=True):
            
            # 1. Feature Engineering
            df_feats = build_scm_features(target_df, technique, horizon_label)
            
            # 2. Train XGBoost
            features = [c for c in df_feats.columns if c.startswith('feat_') or c in ['month', 'dayofweek']]
            X = df_feats[features]
            y = df_feats['order_qty']
            
            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, objective="reg:squarederror")
            model.fit(X, y)
            
            # 3. Predict Future
            h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
            last_date = target_df['Date'].max()
            future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
            
            # For multi-step future, we use the last known features as a stable baseline
            last_features = X.iloc[[-1]]
            preds = model.predict(pd.concat([last_features]*len(future_dates)))
            
            # Specific logic for Ramp Up technique (Apply momentum multiplier)
            if technique == "Ramp Up Evenly":
                preds = [p * (1.02 ** i) for i, p in enumerate(preds)]

            # 4. Results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Historical Demand", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Forecast", line=dict(color="#e67e22", dash='dot')))
            fig.update_layout(title=f"Trend: {item_name} ({technique})", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "Predicted Qty": np.round(preds, 1)})
            st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
