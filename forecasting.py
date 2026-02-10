import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI SETTINGS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide")

# Custom CSS for a professional, interactive look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .step-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-top: 4px solid #00B0F0;
    }
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
    }
    .execute-btn > button {
        width: 100% !important;
        background-color: #007bff !important;
        color: white !important;
        font-weight: bold !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }
    .control-label {
        font-weight: bold;
        color: #1e3d59;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Supply Chain AI Forecast Dashboard")

# --- INPUT SECTION ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("1. Setup Scope & Frequency")
    main_choice = st.radio("Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
    
    c1, c2 = st.columns(2)
    with c1:
        interval = st.selectbox("Forecast Interval", options=["Daily", "Weekly", "Monthly", "Quarterly"])
    with c2:
        if main_choice == "Product Wise":
            sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True)
        else: st.write("")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("2. Intelligence Strategy")
    technique = st.selectbox("AI Base Strategy", ["Historical Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
    
    if technique == "Moving Average":
        n_window = st.number_input("Lookback window", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        ramp_f = st.slider("Growth Multiplier", 1.0, 1.5, 1.05)
    else: st.write("Standard Strategy Applied")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.subheader("3. Data Ingestion")
uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- BASELINE LOGIC ---
def get_baseline(demand, tech):
    if len(demand) == 0: return 0
    if tech == "Moving Average": return np.mean(demand[-7:])
    if tech == "Exponentially": return demand[-1] * 0.3 + np.mean(demand) * 0.7
    return np.mean(demand)

# --- PROCESSING ---
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Total Aggregate"
        else:
            selected = st.selectbox(f"Select Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        execute = st.button("🚀 INITIALIZE AI MODEL")
        st.markdown('</div>', unsafe_allow_html=True)

        if execute or 'model_ready' in st.session_state:
            st.session_state.model_ready = True
            
            # --- RESULTS SECTION ---
            st.divider()
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader(f"📈 Predictive Trend Analysis: {item_name}")
            
            # LIVE CONTROLS FOR TREND GRAPH
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                # The user can now change 15 days, 3 months, etc. instantly here
                horizon_val = st.slider("Adjust Forecast Length (Periods)", 1, 100, 30)
            with col_ctrl2:
                horizon_type = st.radio("Unit", ["Days", "Weeks", "Months"], horizontal=True)

            # --- AI LOGIC ---
            history = target_df['qty'].tolist()
            base_val = get_baseline(history, technique)
            
            # Train model once on click
            target_df['m'] = target_df['Date'].dt.month
            target_df['d'] = target_df['Date'].dt.dayofweek
            target_df['residual'] = target_df['qty'] - base_val
            
            model = XGBRegressor(n_estimators=50)
            model.fit(target_df[['m', 'd']], target_df['residual'])
            
            # Generate Future dates based on LIVE SLIDER
            last_date = target_df['Date'].max()
            if horizon_type == "Days": freq_str = "D"
            elif horizon_type == "Weeks": freq_str = "W"
            else: freq_str = "M"
            
            future_dates = pd.date_range(start=last_date, periods=horizon_val + 1, freq=freq_str)[1:]
            
            # Forecast Prediction
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['m'], f_df['d'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            preds = base_val + model.predict(f_df[['m', 'd']])
            preds = np.maximum(preds, 0)

            if technique == "Ramp Up Evenly":
                preds = [p * (ramp_f ** i) for i, p in enumerate(preds, 1)]

            # Visuals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Historical Data", line=dict(color="#1e3d59")))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Dynamic AI Forecast", line=dict(color="#FF8C00", width=3, dash='dot')))
            
            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0),
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary Metrics
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Current Baseline", f"{base_val:.2f}")
            c_m2.metric("Forecast Peak", f"{np.max(preds):.2f}")
            c_m3.metric("Total Expected Volume", f"{np.sum(preds):.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Please check your file format. Error: {e}")

else:
    st.info("Please upload a file to begin.")
