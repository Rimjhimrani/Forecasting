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

# --- STEP 2: TIMELINE (UPDATED WITH QUANTITY) ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">2</div>Timeline Configuration</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    interval = st.selectbox("Forecast Interval (Frequency)", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_b:
    horizon_unit = st.selectbox("Forecast Horizon Unit", ["Days", "Weeks", "Months", "Years"], index=2)
with col_c:
    horizon_value = st.number_input(f"Number of {horizon_unit} to Forecast", min_value=1, value=1, step=1)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUES ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">3</div>AI Strategy & Technique</div>', unsafe_allow_html=True)
col_d, col_e = st.columns(2)
with col_d:
    technique = st.selectbox("Select Statistical Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
with col_e:
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
        tech_params['ramp_factor'] = st.number_input("Growth/Ramp Factor (Multiplier per step)", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">4</div>Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel (Wide format: Dates as Columns)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- BASELINE CALCULATION ---
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

# --- EXECUTION LOGIC ---
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

        # Data Selection
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate Sum"
        else:
            selected_items = df_long[id_col].unique()
            selected = st.selectbox(f"Select Target {sub_choice}", selected_items)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Resampling based on Interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # Forecast Run Button
        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        if st.button("🚀 EXECUTE HYBRID AI FORECAST"):
            with st.spinner('Generating Insights...'):
                history = target_df['qty'].tolist()
                excel_base = calculate_excel_baseline(history, technique, tech_params)
                
                # AI Modeling (Residual learning)
                target_df['month'] = target_df['Date'].dt.month
                target_df['dow'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - excel_base
                
                model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
                model.fit(target_df[['month', 'dow']], target_df['diff'])
                
                # --- NEW HORIZON CALCULATION ---
                # Map units to Pandas Timedelta/Offsets
                unit_map = {"Days": "D", "Weeks": "W", "Months": "M", "Years": "Y"}
                last_date = target_df['Date'].max()
                
                # Create future date range based on Interval frequency
                # We extend the range until it covers the 'horizon_value' of 'horizon_unit'
                if horizon_unit == "Days": end_date = last_date + pd.Timedelta(days=horizon_value)
                elif horizon_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=horizon_value)
                elif horizon_unit == "Months": end_date = last_date + pd.DateOffset(months=horizon_value)
                else: end_date = last_date + pd.DateOffset(years=horizon_value)

                future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
                
                # Safety check if range is too small
                if len(future_dates) == 0:
                    future_dates = pd.date_range(start=last_date, periods=horizon_value + 1, freq=res_map[interval])[1:]

                f_df = pd.DataFrame({'Date': future_dates})
                f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                ai_wiggles = model.predict(f_df[['month', 'dow']])
                
                final_preds = np.maximum(excel_base + ai_wiggles, 0)
                
                # Apply Ramp Up growth factor if selected
                if technique == "Ramp Up Evenly":
                    rf = tech_params.get('ramp_factor', 1.05)
                    final_preds = [p * (rf ** i) for i, p in enumerate(final_preds, 1)]

                # --- RESULTS ---
                st.success(f"Forecast for {len(future_dates)} steps generated!")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Scope", item_name)
                m2.metric("Stat Baseline", f"{excel_base:.2f}")
                m3.metric("Avg Forecast", f"{np.mean(final_preds):.2f}")

                # Graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="History", line=dict(color="#1e3d59", width=2)))
                fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name="AI Forecast", line=dict(color="#FF8C00", width=3, dash='dot')))
                fig.update_layout(
                    title=f"Predictive Trend Analysis: {item_name}", 
                    template="plotly_white", 
                    hovermode="x unified",
                    xaxis_title="Timeline",
                    yaxis_title="Quantity"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Data View
                with st.expander("View Detailed Forecast Data Table"):
                    date_fmt = '%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'
                    res_df = pd.DataFrame({
                        "Forecast Date": future_dates.strftime(date_fmt), 
                        "Predicted Qty": np.round(final_preds, 2)
                    })
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Processing Error: {e}")
