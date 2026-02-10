import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI SETTINGS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide")

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main { background-color: #f4f7f6; }
    .step-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border-left: 5px solid #00B0F0;
    }
    .step-header {
        color: #1e3d59;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .step-number {
        background-color: #00B0F0;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 12px;
        font-size: 16px;
    }
    .execute-btn > button {
        width: 100% !important;
        background-color: #00B050 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 18px !important;
    }
    .trend-control-card {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #90caf9;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI-Powered Supply Chain Precision Forecast")

# --- STEP 1: SCOPE ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">1</div>Choose Forecasting Scope</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Primary Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Specific Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: TIMELINE (HORIZON KEPT HERE) ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">2</div>Timeline Configuration</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_b:
    horizon_label = st.selectbox("Default Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUES ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">3</div>AI Strategy & Technique</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)
with col_c:
    technique = st.selectbox("Select Statistical Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
with col_d:
    if technique == "Weightage Average":
        w_mode = st.radio("Weight Mode", ["Automated", "Manual"], horizontal=True)
        if w_mode == "Manual":
            w_in = st.text_input("Enter weights (comma separated)", "0.2, 0.3, 0.5")
            try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
            except: st.error("Invalid weight format")
        else: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">4</div>Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC ---
def calculate_excel_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.33, 0.33, 0.34]))
        n = len(w)
        return np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
    elif tech == "Ramp Up Evenly":
        d_slice = demand[-7:] if len(demand) >= 7 else demand
        weights = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, weights) / weights.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate Sum"
        else:
            selected = st.selectbox(f"Select {sub_choice}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        if st.button("🚀 EXECUTE HYBRID AI FORECAST"):
            st.session_state.execute_clicked = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('execute_clicked', False):
            st.divider()
            
            # --- TREND CHART CONTROLS (THE DROP BOX YOU ASKED FOR) ---
            st.markdown('<div class="trend-control-card">', unsafe_allow_html=True)
            st.write("🔄 **Change Forecast View Instantly:**")
            col_z1, col_z2 = st.columns(2)
            with col_z1:
                dynamic_h_val = st.number_input("How many units?", min_value=1, value=30)
            with col_z2:
                dynamic_h_unit = st.selectbox("Unit", ["Days", "Weeks", "Months", "Years", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculation
            history = target_df['qty'].tolist()
            excel_base = calculate_excel_baseline(history, technique, tech_params)
            
            # AI logic
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            last_date = target_df['Date'].max()
            
            # Determine which horizon to use
            if dynamic_h_unit == "Original Selection":
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                end_date = last_date + pd.Timedelta(days=h_map[horizon_label])
            elif dynamic_h_unit == "Days": end_date = last_date + pd.Timedelta(days=dynamic_h_val)
            elif dynamic_h_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=dynamic_h_val)
            elif dynamic_h_unit == "Months": end_date = last_date + pd.DateOffset(months=dynamic_h_val)
            elif dynamic_h_unit == "Years": end_date = last_date + pd.DateOffset(years=dynamic_h_val)

            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            
            # Safety for empty range
            if len(future_dates) == 0:
                future_dates = pd.date_range(start=last_date, periods=dynamic_h_val+1, freq=res_map[interval])[1:]

            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_wiggles = model.predict(f_df[['month', 'dow']])
            final_preds = np.maximum(excel_base + ai_wiggles, 0)

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="History", line=dict(color="#1e3d59")))
            fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name="AI Forecast", line=dict(color="#FF8C00", dash='dot', width=3)))
            fig.update_layout(title=f"Trend Projection for {item_name}", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.subheader("📋 Forecasted Timeline")
            date_fmt = '%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'
            res_df = pd.DataFrame({"Date": future_dates.strftime(date_fmt), "Forecast Qty": np.round(final_preds, 2)})
            st.dataframe(res_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
