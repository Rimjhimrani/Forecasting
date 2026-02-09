import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# --- UI Setup ---
st.set_page_config(page_title="Forecasting System", layout="centered")

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

st.title("📦 Comprehensive Order Forecasting")

# --- STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- STEP 2: SELECT TIMELINE ---
st.markdown('<div class="step-header">STEP 2: Select Timeline</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    interval = st.selectbox("Forecast Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    horizon_label = st.selectbox("Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- STEP 3: CHOOSE FORECAST TECHNIQUE ---
st.markdown('<div class="step-header">STEP 3: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Method", [
    "XGBoost AI (Recommended)",
    "Historical Average", 
    "Moving Average", 
    "Weightage Average", 
    "Ramp Up Evenly", 
    "Exponentially"
])

# Logic for Technique-specific parameters
tech_params = {}
if technique == "Weightage Average":
    w_mode = st.radio("Weight Calculation", ["Automated (Even)", "Manual Entry"], horizontal=True)
    if w_mode == "Manual Entry":
        w_input = st.text_input("Enter weights (comma separated, e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
        tech_params['weights'] = [float(x.strip()) for x in w_input.split(',')]
    else:
        tech_params['window'] = st.number_input("Lookback Window for Even Weights", 2, 12, 3)

elif technique == "Moving Average":
    tech_params['window'] = st.number_input("Moving Average Window", 2, 24, 3)

elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Interval Ramp up Factor (e.g., 1.05 for 5% growth)", 1.0, 2.0, 1.05, 0.01)

# --- STEP 4: UPLOAD DATA ---
st.markdown('<div class="step-header">STEP 4: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

# --- CORE ENGINE FUNCTIONS ---
@st.cache_data(show_spinner=False)
def flexible_transform_data(file_content, is_csv):
    raw = pd.read_csv(file_content) if is_csv else pd.read_excel(file_content)
    raw.columns = raw.columns.astype(str).str.strip()
    id_col = raw.columns[0]
    valid_date_cols = [col for col in raw.columns[1:] if pd.notnull(pd.to_datetime(col, errors='coerce', dayfirst=True))]
    df_long = raw.melt(id_vars=[id_col], value_vars=valid_date_cols, var_name='RawDate', value_name='order_qty')
    df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
    df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
    return df_long.dropna(subset=['Date']).sort_values('Date'), id_col

def run_forecast_logic(data, horizon_days, interval_type, tech, params):
    df = data.copy()
    history = df['order_qty'].values
    
    # Calculate steps
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    freq = res_map[interval_type]
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=horizon_days), freq=freq)[1:]
    if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=freq)[1:]
    steps = len(future_dates)

    preds = []
    
    # 1. XGBoost AI
    if tech == "XGBoost AI (Recommended)":
        df['h'], df['d'], df['m'], df['y'], df['dw'] = df['Date'].dt.hour, df['Date'].dt.day, df['Date'].dt.month, df['Date'].dt.year, df['Date'].dt.dayofweek
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(df[['h', 'd', 'm', 'y', 'dw']], df['order_qty'])
        f_df = pd.DataFrame({'Date': future_dates})
        f_df['h'], f_df['d'], f_df['m'], f_df['y'], f_df['dw'] = f_df['Date'].dt.hour, f_df['Date'].dt.day, f_df['Date'].dt.month, f_df['Date'].dt.year, f_df['Date'].dt.dayofweek
        preds = model.predict(f_df[['h', 'd', 'm', 'y', 'dw']])

    # 2. Historical Average
    elif tech == "Historical Average":
        preds = [np.mean(history)] * steps

    # 3. Moving Average
    elif tech == "Moving Average":
        w = params['window']
        val = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
        preds = [val] * steps

    # 4. Weightage Average
    elif tech == "Weightage Average":
        if 'weights' in params:
            w = len(params['weights'])
            weights = np.array(params['weights'])
            val = np.dot(history[-w:], weights) / weights.sum() if len(history) >= w else np.mean(history)
        else:
            w = params['window']
            weights = np.ones(w)
            val = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
        preds = [val] * steps

    # 5. Ramp Up Evenly
    elif tech == "Ramp Up Evenly":
        base = history[-1]
        preds = [base * (params['ramp_factor'] ** i) for i in range(1, steps + 1)]

    # 6. Exponentially
    elif tech == "Exponentially":
        model = SimpleExpSmoothing(history).fit(smoothing_level=0.3, optimized=False)
        preds = model.forecast(steps)

    return future_dates, np.maximum(preds, 0)

# --- EXECUTION ---
if uploaded_file:
    try:
        df_long, id_col = flexible_transform_data(uploaded_file, uploaded_file.name.endswith('.csv'))
        
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Aggregate"
        else:
            selected_item = st.selectbox(f"Select {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected_item].copy()
            item_label = str(selected_item)

        target_df = target_df.set_index('Date').resample({"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}[interval]).sum().reset_index()

        st.markdown('<div class="step-header">STEP 5: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
            f_dates, f_values = run_forecast_logic(target_df, h_map[horizon_label], interval, technique, tech_params)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="History", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=f_dates, y=f_values, name=f"Forecast ({technique})", line=dict(color="#e67e22", dash='dot')))
            fig.update_layout(title=f"Trend Analysis: {item_label}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            res_df = pd.DataFrame({"Date": f_dates.strftime('%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'), "Qty": np.round(f_values, 1)})
            st.dataframe(res_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
