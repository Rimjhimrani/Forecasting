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
st.info("XGBoost engine using your specific classical formulas as input features.")

# --- FLOWCHART STEP 1: OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Option", ["Aggregate Wise", "Product Wise"], horizontal=True)
sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Product Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- FLOWCHART STEP 2: INTERVAL ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval</div>', unsafe_allow_html=True)
interval = st.selectbox("Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)

# --- FLOWCHART STEP 3: HORIZON ---
st.markdown('<div class="step-header">STEP 3: Select Forecast Horizon</div>', unsafe_allow_html=True)
horizon_label = st.selectbox("Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: TECHNIQUE ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Techniques</div>', unsafe_allow_html=True)
technique = st.selectbox("Technique", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
if technique == "Weightage Average":
    w_mode = st.radio("Weight Mode", ["Automated", "Manually entering of weights"], horizontal=True)
    if w_mode == "Manually entering of weights":
        w_in = st.text_input("Enter weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
        tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
    else:
        tech_params['weights'] = np.array([0.2, 0.3, 0.5]) # Default automated

elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Window (n)", 2, 30, 7)

elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Manually entering of Interval Ramp up Factor", 1.0, 2.0, 1.05, 0.01)

# --- YOUR FORMULAS INTEGRATED INTO XGBOOST FEATURES ---
def build_xgboost_features(df, tech, params):
    """Uses your exact formulas to create input signals for XGBoost"""
    df = df.rename(columns={'order_qty': 'demand'})
    
    # 1. Historical Average
    df["feat_hist_avg"] = df["demand"].expanding().mean().shift(1)
    
    # 2. Moving Averages
    for n in [7, 14, 28]:
        df[f"feat_ma_{n}"] = df["demand"].rolling(window=n, min_periods=1).mean().shift(1)
        
    # 3. Weighted Average
    def weighted_avg_calc(series, weights):
        if len(series) < len(weights): return np.mean(series)
        return np.dot(series, weights) / np.sum(weights)
    
    w = params.get('weights', np.array([0.2, 0.3, 0.5]))
    df["feat_weighted_avg"] = df["demand"].rolling(window=len(w), min_periods=1).apply(lambda x: weighted_avg_calc(x, w), raw=True).shift(1)
    
    # 4. Ramp Up
    def ramp_up_calc(series):
        n = len(series)
        weights = np.arange(1, n + 1)
        return np.dot(series, weights) / weights.sum()
    
    for n in [7, 14]:
        df[f"feat_ramp_up_{n}"] = df["demand"].rolling(window=n, min_periods=1).apply(ramp_up_calc, raw=True).shift(1)
        
    # 5. Exponential Smoothing
    def exp_smooth_calc(series, alpha):
        if len(series) == 0: return 0
        result = [series.iloc[0]]
        for d in series.iloc[1:]:
            result.append(alpha * d + (1 - alpha) * result[-1])
        return pd.Series(result, index=series.index)

    for alpha in [0.2, 0.4, 0.6]:
        df[f"feat_exp_{alpha}"] = exp_smooth_calc(df["demand"], alpha).shift(1)

    # 6. Time Features
    df['feat_month'] = df['Date'].dt.month
    df['feat_dow'] = df['Date'].dt.dayofweek
    
    return df.fillna(method='bfill').fillna(0)

# --- STEP 5: UPLOAD DATA ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # Load and Transform Wide Format
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        
        valid_date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=valid_date_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_name = "Aggregate Total"
        else:
            options = df_long[id_col].unique()
            selected = st.selectbox(f"Select from {id_col}", options)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Resample
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: EXECUTION ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast and Trend Analysis", use_container_width=True):
            with st.spinner('Calculating XGBoost Forecast...'):
                # 1. Feature Engineering
                df_final = build_xgboost_features(target_df, technique, tech_params)
                
                # 2. Train XGBoost
                X_cols = [c for c in df_final.columns if c.startswith('feat_')]
                X = df_final[X_cols]
                y = df_final['demand']
                
                model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, objective="reg:squarederror")
                model.fit(X, y)
                
                # 3. Setup Future
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
                
                if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                # Prediction (using last historical feature state)
                last_feat_row = X.tail(1)
                preds = model.predict(pd.concat([last_feat_row]*len(future_dates)))
                
                # Apply Ramp Factor if selected
                if technique == "Ramp Up Evenly":
                    preds = [p * (tech_params['ramp_factor'] ** i) for i, p in enumerate(preds, 1)]

                # 4. Results Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Past data", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"AI Forecast ({technique})", line=dict(color="#e67e22", dash='dot')))
                fig.update_layout(title=f"Trend Analysis: {item_name}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'), "Predicted Qty": np.round(preds, 1)})
                st.dataframe(res_df, use_container_width=True)
                st.success("END OF PROCESS")

    except Exception as e:
        st.error(f"Error: {e}")
