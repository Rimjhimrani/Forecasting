import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI SETTINGS & FLOWCHART STYLING ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide")

st.markdown("""
    <style>
    .step-header {
        background-color: #00B0F0;
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        font-size: 20px;
        border: 2px solid #000;
    }
    .start-label { background-color: #FFFF00; color: black; padding: 5px 15px; border: 2px solid #000; font-weight: bold; width: fit-content; margin: auto; margin-bottom: 10px; }
    .end-label { background-color: #00B050; color: white; padding: 5px 15px; border: 2px solid #000; font-weight: bold; width: fit-content; margin: auto; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="start-label">START</div>', unsafe_allow_html=True)
st.title("📊 Supply Chain Forecasting Flow")

# --- STEP 1: CHOOSE OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Primary Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Product Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- STEP 2: INTERVAL ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval</div>', unsafe_allow_html=True)
interval = st.select_slider("Select Frequency", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], value="Daily")

# --- STEP 3: HORIZON ---
st.markdown('<div class="step-header">STEP 3: Select Forecast Horizon</div>', unsafe_allow_html=True)
horizon_label = st.selectbox("Forecast Horizon (Future Length)", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- STEP 4: TECHNIQUES ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Techniques</div>', unsafe_allow_html=True)
technique = st.selectbox("AI Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
if technique == "Weightage Average":
    w_mode = st.radio("Weight Selection", ["Automated (Even)", "Manual Entry"], horizontal=True)
    if w_mode == "Manual Entry":
        w_in = st.text_input("Enter weights (e.g. 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
        tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
    else: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Manual Ramp Up Factor (Growth)", 1.0, 2.0, 1.05, 0.01)
elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)

# --- STEP 5: UPLOAD ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV (Dates as Columns)", type=['xlsx', 'csv'])

# --- BASELINE CALCULATION LOGIC ---
def calculate_excel_baseline(demand, tech, params):
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average": 
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.33, 0.33, 0.34]))
        n = len(w)
        return np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
    elif tech == "Ramp Up Evenly":
        d_slice = demand[-7:] # Default ramp-up window
        weights = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, weights) / weights.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

# --- EXECUTION ---
if uploaded_file:
    try:
        # 1. Melt Wide to Long
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        # 2. Filter Path
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate"
        else:
            selected = st.selectbox(f"Select {sub_choice}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- STEP 6: AI GENERATION ---
        st.markdown('<div class="step-header" style="background-color:#FFFF00; color:black;">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 EXECUTE HYBRID AI FORECAST"):
            with st.spinner('Applying Hybrid AI logic...'):
                history = target_df['qty'].tolist()
                
                # 1. Excel Baseline Calculation
                excel_base = calculate_excel_baseline(history, technique, tech_params)
                
                # 2. Create AI Target (Residual vs Baseline)
                target_df['month'] = target_df['Date'].dt.month
                target_df['dow'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - excel_base
                
                # 3. Train AI (XGBoost)
                model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
                model.fit(target_df[['month', 'dow']], target_df['diff'])
                
                # 4. Predict Future Adjustment
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
                if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                f_df = pd.DataFrame({'Date': future_dates})
                f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                
                ai_wiggles = model.predict(f_df[['month', 'dow']])
                
                # 5. Final Forecast (Baseline + AI pattern)
                final_preds = excel_base + ai_wiggles
                final_preds = np.maximum(final_preds, 0) # Safety check
                
                if technique == "Ramp Up Evenly":
                    final_preds = [p * (tech_params['ramp_factor'] ** i) for i, p in enumerate(final_preds, 1)]

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="History", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name="AI Forecast", line=dict(color="#FF8C00", dash='dot', width=3)))
                fig.update_layout(title=f"Trend Analysis for {item_name} (Hybrid AI Logic)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📋 Predicted Data Results")
                date_fmt = '%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'
                res_df = pd.DataFrame({"Date": future_dates.strftime(date_fmt), "Forecast Qty": np.round(final_preds, 2)})
                st.dataframe(res_df, use_container_width=True)
                
                st.write(f"**Calculated Excel Baseline:** {round(excel_base, 2)}")
                st.markdown('<div class="end-label">END</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in process: {e}")
else:
    st.info("Upload the wide-format data file in Step 5 to activate Step 6.")
