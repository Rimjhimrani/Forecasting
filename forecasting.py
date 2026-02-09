import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI Setup ---
st.set_page_config(page_title="Precision AI Forecasting", layout="centered")

st.markdown("""
    <style>
    .step-header {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin-bottom: 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📦 Precision Order Forecasting")
st.info("System optimized for your Wide-Format data. Powered by Hybrid XGBoost AI.")

# --- FLOWCHART STEPS 1-4 ---
st.markdown('<div class="step-header">STEP 1-4: Configuration</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    main_choice = st.radio("Option", ["Aggregate Wise", "Product Wise"], horizontal=True)
    interval = st.selectbox("Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    
with col_b:
    sub_choice = st.radio("Product Level", ["Model Wise", "Part No Wise"], horizontal=True) if main_choice == "Product Wise" else None
    horizon_label = st.selectbox("Horizon (Future Length)", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

technique = st.selectbox("Forecast Technique", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

# Technique Specific Inputs
tech_params = {}
if technique == "Weightage Average":
    w_in = st.text_input("Manual Weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
    tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Window Size (n)", 2, 30, 7)
elif technique == "Ramp Up Evenly":
    tech_params['n'] = 7
    tech_params['factor'] = st.number_input("Growth Multiplier Factor", 1.0, 2.0, 1.05)
elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)

# --- HYBRID CALCULATION ENGINE ---
def calculate_excel_baseline(demand, tech, params):
    """Calculates the exact mathematical baseline from your Excel logic"""
    if tech == "Historical Average":
        return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:])
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.2, 0.3, 0.5]))
        n = len(w)
        if len(demand) < n: return np.mean(demand)
        return np.dot(demand[-n:], w) / np.sum(w)
    elif tech == "Ramp Up Evenly":
        n = params.get('n', 7)
        d_slice = demand[-n:]
        w = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, w) / w.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]:
            forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

# --- STEP 5: UPLOAD & PROCESS ---
st.markdown('<div class="step-header">STEP 5: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your Wide-Format file (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # Load and Transform based on your image
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0] # Usually 'MODEL'
        
        # Melt columns to convert dates to rows
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='Date', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        # Filtering
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate Total"
        else:
            options = df_long[id_col].unique()
            selected = st.selectbox(f"Select from {id_col}", options)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Consolidate to selected interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: EXECUTION ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            history = target_df['qty'].tolist()
            
            # 1. Get Formula Baseline (e.g. 3.4 for 3W model)
            excel_base = calculate_excel_baseline(history, technique, tech_params)
            
            # 2. Train XGBoost to learn pattern wiggles (residuals)
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - np.mean(history) # Learn variation from mean
            
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            # 3. Predict Future
            h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
            last_date = target_df['Date'].max()
            future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
            
            if len(future_dates) == 0:
                future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'] = f_df['Date'].dt.month
            f_df['dow'] = f_df['Date'].dt.dayofweek
            
            # AI predicts the "wiggle" around the excel base
            ai_wiggles = model.predict(f_df[['month', 'dow']])
            
            # Final Forecast = Excel Baseline + AI pattern
            final_preds = excel_base + ai_wiggles
            
            # Special logic for Ramp Up
            if technique == "Ramp Up Evenly":
                f = tech_params.get('factor', 1.05)
                final_preds = [p * (f ** i) for i, p in enumerate(final_preds, 1)]

            # 4. Results Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Actual History", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name=f"AI Forecast (Base: {round(excel_base,2)})", line=dict(color="#e67e22", dash='dot')))
            fig.update_layout(title=f"Trend Analysis: {item_name} (Hybrid XGBoost)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Format and Table Display
            date_fmt = '%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'
            res_df = pd.DataFrame({
                "Date": future_dates.strftime(date_fmt), 
                "Predicted Qty": np.round(final_preds, 2)
            })
            st.dataframe(res_df, use_container_width=True)
            st.write(f"**Excel formula baseline for this period:** {round(excel_base, 2)}")
            st.success("Analysis Complete (End of Flowchart)")

    except Exception as e:
        st.error(f"Error processing data: {e}")
