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
        border-left: 5px solid #28a745;
        margin-bottom: 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Precision AI Forecasting")
st.info("Aligned with Excel Formulas + XGBoost Seasonal Intelligence.")

# --- FLOWCHART STEPS 1-3 ---
st.markdown('<div class="step-header">STEP 1-3: Setup Selection</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_option = st.radio("Option", ["Aggregate Wise", "Product Wise"], horizontal=True)
    interval = st.selectbox("Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    sub_option = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True) if main_option == "Product Wise" else None
    horizon_label = st.selectbox("Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: TECHNIQUE ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Technique", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
if technique == "Weightage Average":
    w_in = st.text_input("Manual Weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
    tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Window Size (n)", 2, 30, 7)
elif technique == "Ramp Up Evenly":
    tech_params['n'] = 7
    tech_params['factor'] = st.number_input("Ramp up Factor", 1.0, 2.0, 1.05)
elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Alpha", 0.01, 1.0, 0.3)

# --- THE HYBRID ENGINE (Excel Math + AI Pattern) ---
def get_formula_baseline(demand, tech, params):
    """Calculates the exact Excel-style baseline"""
    if tech == "Historical Average":
        return sum(demand) / len(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return sum(demand[-n:]) / n
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.2, 0.3, 0.5]))
        n = len(w)
        d_slice = demand[-n:]
        return np.dot(d_slice, w) / np.sum(w) if len(d_slice) == n else np.mean(demand)
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

# --- STEP 5: UPLOAD DATA ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        if main_option == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_name = "Aggregate"
        else:
            selected = st.selectbox(f"Select from {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: EXECUTION ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            history = target_df['order_qty'].tolist()
            
            # 1. GET EXCEL BASELINE (e.g., your 3.4)
            excel_base = get_formula_baseline(history, technique, tech_params)
            
            # 2. USE XGBOOST TO LEARN SEASONAL DEVIATION
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            # We train on 'deviation from average' so AI only learns the 'shape'
            target_df['deviation'] = target_df['order_qty'] - np.mean(history)
            
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['deviation'])
            
            # 3. GENERATE FUTURE
            h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
            last_date = target_df['Date'].max()
            future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
            if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

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

            # 4. RESULTS
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Actual History", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name=f"AI Forecast (Base: {round(excel_base,2)})", line=dict(color="#e67e22", dash='dot')))
            fig.update_layout(title=f"Trend Analysis: {item_name} (Hybrid XGBoost)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "Predicted Qty": np.round(final_preds, 2)})
            st.dataframe(res_df, use_container_width=True)
            st.write(f"**Excel formula baseline for this period:** {round(excel_base, 2)}")

    except Exception as e:
        st.error(f"Error: {e}")
