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
st.info("Uses classical formulas (Historical Avg, MA, etc.) as features for the XGBoost AI engine.")

# --- FLOWCHART STEPS 1-3 ---
st.markdown('<div class="step-header">STEPS 1-3: Configuration</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
    interval = st.selectbox("Forecast Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    
with col2:
    sub_choice = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True) if main_choice == "Product Wise" else None
    horizon_label = st.selectbox("Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: CHOOSE TECHNIQUE ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Method (AI Strategy)", 
                         ["Historical Average", "Moving Average", "Weightage Average", "Ramp Up Evenly", "Exponentially"])

# --- CORE FEATURE ENGINE ---
def build_scm_features(df, total_rows):
    """Safely calculates classical formulas as features for XGBoost"""
    data = df.copy()
    
    # Feature 1: Historical Average
    data['feat_hist_avg'] = data['order_qty'].expanding().mean().shift(1)
    
    # Feature 2: Moving Averages (Safe windows)
    windows = [3, 7, 14]
    for n in windows:
        if total_rows > n:
            data[f'feat_ma_{n}'] = data['order_qty'].rolling(window=n).mean().shift(1)
        else:
            # Fallback for small data
            data[f'feat_ma_{n}'] = data['order_qty'].expanding().mean().shift(1)
            
    # Feature 3: Exponential Smoothing Signal
    data['feat_exp'] = data['order_qty'].ewm(alpha=0.3).mean().shift(1)
    
    # Time signals
    data['month'] = data['Date'].dt.month
    data['dayofweek'] = data['Date'].dt.dayofweek
    data['hour'] = data['Date'].dt.hour
    
    # Fill any gaps created by shifting
    return data.fillna(method='bfill').fillna(0)

# --- STEP 5: UPLOAD & PROCESS ---
st.markdown('<div class="step-header">STEP 5: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV (Dates as Columns)", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # Load and Melt Wide to Long
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        
        valid_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=valid_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter Logic
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_name = "Total Aggregate"
        else:
            options = df_long[id_col].unique()
            selected = st.selectbox(f"Select item from {id_col}", options)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Resample to Interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: EXECUTION ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Run Analysis", use_container_width=True):
            if len(target_df) < 3:
                st.error("Not enough historical data points (minimum 3 required).")
            else:
                with st.spinner('AI analyzing patterns...'):
                    # 1. Feature Engineering
                    df_feats = build_scm_features(target_df, len(target_df))
                    
                    # 2. Train XGBoost
                    feature_cols = [c for c in df_feats.columns if c.startswith('feat_') or c in ['month', 'dayofweek', 'hour']]
                    X = df_feats[feature_cols]
                    y = df_feats['order_qty']
                    
                    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
                    model.fit(X, y)
                    
                    # 3. Predict Future
                    h_days = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                    last_date = target_df['Date'].max()
                    future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_days[horizon_label]), freq=res_map[interval])[1:]
                    
                    if len(future_dates) == 0:
                        future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                    # Use the most recent features as the prediction baseline
                    # FIX: Use .tail(1) to avoid index out-of-bounds error
                    last_row_features = X.tail(1)
                    preds = model.predict(pd.concat([last_row_features]*len(future_dates)))
                    
                    # Technique specific adjustments (Ramp up growth or exponential decay)
                    if technique == "Ramp Up Evenly":
                        preds = [p * (1.03 ** i) for i, p in enumerate(preds)]
                    elif technique == "Exponentially":
                        preds = [p * (0.98 ** i) for i, p in enumerate(preds)]

                    # 4. Results Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Historical Data", line=dict(color="#2c3e50")))
                    fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"AI Forecast ({technique})", line=dict(color="#e67e22", dash='dot')))
                    fig.update_layout(title=f"Trend Analysis: {item_name}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    date_fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                    res_df = pd.DataFrame({"Date": future_dates.strftime(date_fmt), "Predicted Qty": np.round(preds, 1)})
                    st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
