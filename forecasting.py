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

# --- FLOWCHART STEPS 1-3 ---
st.markdown('<div class="step-header">STEP 1-3: Selection, Interval & Horizon</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_option = st.radio("Option", ["Aggregate Wise", "Product Wise"], horizontal=True)
    interval = st.selectbox("Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    sub_option = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True) if main_option == "Product Wise" else None
    horizon_label = st.selectbox("Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: CHOOSE TECHNIQUE ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Techniques</div>', unsafe_allow_html=True)
technique = st.selectbox("Technique", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
if technique == "Weightage Average":
    w_in = st.text_input("Manual Weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
    tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Window Size (n)", 2, 30, 7)
elif technique == "Ramp Up Evenly":
    tech_params['n'] = [7, 14] # Standard windows for ramp up signal
    tech_params['factor'] = st.number_input("Ramp up Factor", 1.0, 2.0, 1.05)
elif technique == "Exponentially":
    tech_params['alphas'] = [0.2, 0.4, 0.6]

# --- FEATURE ENGINE (Using your exact formulas) ---
def generate_features(df, tech, params):
    df = df.rename(columns={'order_qty': 'demand'})
    
    # 1. ALWAYS add Time Features (Month and Day of Week)
    df['feat_month'] = df['Date'].dt.month
    df['feat_dow'] = df['Date'].dt.dayofweek
    df['feat_hour'] = df['Date'].dt.hour
    
    # 2. Add technique-specific signal features
    if tech == "Historical Average":
        df["feat_signal"] = df["demand"].expanding().mean().shift(1)
        
    elif tech == "Moving Average":
        n = params.get('n', 7)
        df["feat_signal"] = df["demand"].rolling(window=n, min_periods=1).mean().shift(1)
        
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.2, 0.3, 0.5]))
        df["feat_signal"] = df["demand"].rolling(window=len(w), min_periods=1).apply(
            lambda x: np.dot(x, w)/np.sum(w) if len(x)==len(w) else np.mean(x), raw=True).shift(1)
            
    elif tech == "Ramp Up Evenly":
        def ramp_calc(s):
            weights = np.arange(1, len(s) + 1)
            return np.dot(s, weights) / weights.sum()
        df["feat_signal"] = df["demand"].rolling(window=7, min_periods=1).apply(ramp_calc, raw=True).shift(1)
        
    elif tech == "Exponentially":
        df["feat_signal"] = df["demand"].ewm(alpha=0.4).mean().shift(1)

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
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter
        if main_option == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_name = "Aggregate"
        else:
            selected = st.selectbox(f"Select from {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Resample
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: GENERATE ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            with st.spinner('AI processing specific technique...'):
                # 1. Feature Engineering (TECHNIQUE SPECIFIC)
                df_final = generate_features(target_df, technique, tech_params)
                
                # 2. Train XGBoost on these specific features
                features = ['feat_month', 'feat_dow', 'feat_hour', 'feat_signal']
                X = df_final[features]
                y = df_final['demand']
                
                model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
                model.fit(X, y)
                
                # 3. Predict Future
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
                
                if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                # Create future features (Dynamic time features + carried signal)
                f_df = pd.DataFrame({'Date': future_dates})
                f_df['feat_month'] = f_df['Date'].dt.month
                f_df['feat_dow'] = f_df['Date'].dt.dayofweek
                f_df['feat_hour'] = f_df['Date'].dt.hour
                f_df['feat_signal'] = X['feat_signal'].iloc[-1] # Carry over last signal
                
                preds = model.predict(f_df[features])
                
                # Post-processing for Ramp Up logic
                if technique == "Ramp Up Evenly":
                    f = tech_params.get('factor', 1.05)
                    preds = [p * (f ** i) for i, p in enumerate(preds, 1)]

                # 4. Results
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="History", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"AI Forecast ({technique})", line=dict(color="#e67e22", dash='dot')))
                fig.update_layout(title=f"Trend Analysis: {item_name} via {technique}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'), "Predicted Qty": np.round(preds, 1)})
                st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
