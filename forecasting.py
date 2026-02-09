import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

st.title("📦 Intelligent Order Forecasting System")

# --- FLOWCHART STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-header">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- FLOWCHART STEP 2: SELECT FORECAST INTERVAL ---
st.markdown('<div class="step-header">STEP 2: Select Forecast Interval</div>', unsafe_allow_html=True)
interval = st.selectbox("Frequency", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)

# --- FLOWCHART STEP 3: SELECT FORECAST HORIZON ---
st.markdown('<div class="step-header">STEP 3: Select Forecast Horizon</div>', unsafe_allow_html=True)
horizon_label = st.selectbox("Prediction Length", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: CHOOSE FORECAST TECHNIQUES ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Method", [
    "Historical Average", 
    "Weightage Average", 
    "Moving Average", 
    "Ramp Up Evenly", 
    "Exponentially"
])

# Capture the specific parameters required for your formulas
tech_params = {}
if technique == "Weightage Average":
    w_input = st.text_input("Manually enter weights (comma separated, e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
    tech_params['weights'] = [float(x.strip()) for x in w_input.split(',')]

elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Enter n (Number of periods)", 2, 24, 3)

elif technique == "Ramp Up Evenly":
    tech_params['n'] = st.number_input("Enter n (Window size for ramp up)", 2, 24, 3)

elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Select Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)

# --- FLOWCHART STEP 5: UPLOAD DATA FILE ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

# --- YOUR FORMULAS IMPLEMENTED ---
def historical_average(demand):
    return sum(demand) / len(demand)

def weighted_average(demand, weights):
    # Uses only the last N items where N is the length of weights
    n = len(weights)
    recent_demand = demand[-n:]
    return sum(d * w for d, w in zip(recent_demand, weights)) / sum(weights)

def moving_average(demand, n):
    return sum(demand[-n:]) / n

def ramp_up_evenly(demand, n):
    window = demand[-n:]
    weights = list(range(1, n + 1))
    total_weight = sum(weights)
    return sum(d * w for d, w in zip(window, weights)) / total_weight

def exponential_smoothing(demand, alpha):
    forecast = demand[0]  # initial forecast
    for d in demand[1:]:
        forecast = alpha * d + (1 - alpha) * forecast
    return forecast

# --- DATA PROCESSING & EXECUTION ---
@st.cache_data(show_spinner=False)
def process_data(file_content, is_csv):
    raw = pd.read_csv(file_content) if is_csv else pd.read_excel(file_content)
    raw.columns = raw.columns.astype(str).str.strip()
    id_col = raw.columns[0]
    valid_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
    df_long = raw.melt(id_vars=[id_col], value_vars=valid_cols, var_name='RawDate', value_name='order_qty')
    df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
    df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
    return df_long.dropna(subset=['Date']).sort_values('Date'), id_col

if uploaded_file:
    try:
        df_long, id_col = process_data(uploaded_file, uploaded_file.name.endswith('.csv'))
        
        # Filtering
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Aggregate"
        else:
            selected_item = st.selectbox(f"Select {id_col}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected_item].copy()
            item_label = str(selected_item)

        # Resample to the selected interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()
        
        history = target_df['order_qty'].tolist()

        # FLOWCHART STEP 6: GENERATE FORECAST
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            
            if len(history) < 2:
                st.error("Not enough historical data points.")
            else:
                # Execute specific formula
                if technique == "Historical Average":
                    forecast_val = historical_average(history)
                elif technique == "Weightage Average":
                    forecast_val = weighted_average(history, tech_params['weights'])
                elif technique == "Moving Average":
                    forecast_val = moving_average(history, tech_params['n'])
                elif technique == "Ramp Up Evenly":
                    forecast_val = ramp_up_evenly(history, tech_params['n'])
                elif technique == "Exponentially":
                    forecast_val = exponential_smoothing(history, tech_params['alpha'])

                # Generate Timeline for the table and chart
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                total_days = h_map[horizon_label]
                
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=total_days), freq=res_map[interval])[1:]
                
                if len(future_dates) == 0:
                    future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                # For statistical averages, the predicted value is projected across the horizon
                forecast_results = [forecast_val] * len(future_dates)

                # Show Trend Analysis (Line Chart)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Past data", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast_results, name="Future forecast", line=dict(color="#e67e22", dash='dot')))
                fig.update_layout(title=f"Trend Analysis for {item_label}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Show Data Table
                st.subheader("📋 Forecasted Results Table")
                fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                res_df = pd.DataFrame({"Date": future_dates.strftime(fmt), "Predicted Qty": np.round(forecast_results, 1)})
                st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
