import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

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

st.title("📦 Order Forecasting & Trend Analysis")

# --- FLOWCHART STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- FLOWCHART STEP 2: SELECT FORECAST INTERVAL (Granularity) ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval (Frequency)</div>', unsafe_allow_html=True)
st.caption("Example: 'Weekly forecasts' means one prediction per week.")
interval = st.selectbox("Interval / Spacing", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)

# --- FLOWCHART STEP 3: SELECT FORECAST HORIZON (Total Duration) ---
st.markdown('<div class="step-header">STEP 3: Select Forecast Horizon (Time Length)</div>', unsafe_allow_html=True)
st.caption("Example: '6 months ahead' means the total length of the forecast.")
horizon_label = st.selectbox("How far ahead?", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: CHOOSE FORECAST TECHNIQUES ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Method (All powered by XGBoost AI)", [
    "Historical Average", 
    "Weightage Average", 
    "Moving Average", 
    "Ramp Up Evenly", 
    "Exponentially"
])

# Technique Specific Inputs
tech_params = {}
if technique == "Weightage Average":
    w_mode = st.radio("Weight Calculation", ["Automated (Even Weights)", "Manual Entry of Weights"], horizontal=True)
    if w_mode == "Manual Entry of Weights":
        w_input = st.text_input("Enter weights (e.g., 0.1, 0.2, 0.7)", "0.2, 0.3, 0.5")
        tech_params['weights'] = [float(x.strip()) for x in w_input.split(',')]
    else:
        tech_params['window'] = st.number_input("Lookback window for even weights", 2, 12, 3)

elif technique == "Moving Average":
    tech_params['window'] = st.number_input("Moving Average Lookback Window", 2, 24, 3)

elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Interval Ramp up Factor (Multiplier per period)", 1.0, 5.0, 1.05, 0.01)

# --- FLOWCHART STEP 5: UPLOAD DATA FILE ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

# --- CORE PROCESSING ENGINE ---
@st.cache_data(show_spinner=False)
def process_wide_data(file_content, is_csv):
    raw = pd.read_csv(file_content) if is_csv else pd.read_excel(file_content)
    raw.columns = raw.columns.astype(str).str.strip()
    id_col = raw.columns[0]
    # Filter only columns that can be dates
    valid_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
    df_long = raw.melt(id_vars=[id_col], value_vars=valid_cols, var_name='RawDate', value_name='order_qty')
    df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
    df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
    return df_long.dropna(subset=['Date']).sort_values('Date'), id_col

def run_xgboost_engine(data, horizon_days, interval_type, tech, params):
    df = data.copy()
    
    # Feature Engineering
    df['h'], df['d'], df['m'], df['y'], df['dw'] = df['Date'].dt.hour, df['Date'].dt.day, df['Date'].dt.month, df['Date'].dt.year, df['Date'].dt.dayofweek
    features = ['h', 'd', 'm', 'y', 'dw']
    
    # Technique Logic Integration
    sample_weights = np.ones(len(df))
    if tech == "Weightage Average" and 'weights' in params:
        w_list = params['weights']
        for i, weight in enumerate(reversed(w_list)):
            if (len(df) - 1 - i) >= 0: sample_weights[len(df) - 1 - i] = weight
            
    if tech == "Moving Average":
        df = df.tail(params['window'])
        sample_weights = sample_weights[-len(df):]

    # Training
    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(df[features], df['order_qty'], sample_weight=sample_weights)
    
    # Calculating Steps based on Horizon and Interval
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    freq = res_map[interval_type]
    
    last_date = df['Date'].max()
    end_date = last_date + pd.Timedelta(days=horizon_days)
    
    future_dates = pd.date_range(start=last_date, end=end_date, freq=freq)
    if len(future_dates) <= 1: # Safety: ensures at least one point if duration is short
        future_dates = pd.date_range(start=last_date, periods=2, freq=freq)
    
    future_dates = future_dates[1:] # Predictive part
    
    f_df = pd.DataFrame({'Date': future_dates})
    f_df['h'], f_df['d'], f_df['m'], f_df['y'], f_df['dw'] = f_df['Date'].dt.hour, f_df['Date'].dt.day, f_df['Date'].dt.month, f_df['Date'].dt.year, f_df['Date'].dt.dayofweek
    
    preds = model.predict(f_df[features])
    
    if tech == "Ramp Up Evenly":
        factor = params['ramp_factor']
        preds = [p * (factor ** i) for i, p in enumerate(preds, 1)]
    
    return future_dates, np.maximum(preds, 0)

# --- EXECUTION ---
if uploaded_file:
    try:
        df_long, id_col = process_wide_data(uploaded_file, uploaded_file.name.endswith('.csv'))
        
        # UI Selection
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Aggregate Total"
        else:
            selected_item = st.selectbox(f"Select {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected_item].copy()
            item_label = str(selected_item)

        # Consolidate history to interval frequency
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # --- FLOWCHART STEP 6: GENERATE FORECAST AND TREND ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend Analysis</div>', unsafe_allow_html=True)
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner('XGBoost AI is analyzing...'):
                h_days = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                
                f_dates, f_values = run_xgboost_engine(target_df, h_days[horizon_label], interval, technique, tech_params)

                # Show Trend Analysis (Line Chart)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Past Data", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=f_dates, y=f_values, name=f"Future Forecast ({technique})", line=dict(color="#28a745", dash='dot')))
                fig.update_layout(title=f"Trend Analysis: {item_label}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Display Results Table
                st.subheader("📋 Forecast Summary Table")
                fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                res_df = pd.DataFrame({"Date": f_dates.strftime(fmt), "Predicted Quantity": np.round(f_values, 1)})
                st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
