import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

st.set_page_config(page_title="Advanced Order Forecasting", layout="wide")

# --- UI Header ---
st.title("📊 Order Forecasting System (v2.0)")
st.markdown("---")

# --- 1. Choose an Option (Flowchart: Aggregate vs Product -> Model/Part) ---
col1, col2 = st.columns(2)
with col1:
    main_option = st.radio("Choose Primary Option", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_option = None
if main_option == "Product Wise":
    with col2:
        sub_option = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True)

# --- 2. Select Forecast Interval & Horizon ---
with st.sidebar:
    st.header("Step 1: Configuration")
    interval = st.selectbox("Select Forecast Interval", 
                            ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"])
    
    horizon_map = {
        "Day": 1, "Week": 7, "Month": 30, "Quarter": 90, 
        "Year": 365, "3 years": 365*3, "5 years": 365*5
    }
    horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_map.keys()))

    st.header("Step 2: Technique Settings")
    technique = st.selectbox("Choose Forecast Technique", 
                             ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
    
    # Specific logic based on new flowchart requirements
    params = {}
    if technique == "Weightage Average":
        w_mode = st.radio("Weight Calculation", ["Automated (Even/Linear)", "Manual Entry"])
        if w_mode == "Manual Entry":
            weights_str = st.text_input("Enter weights (comma separated, e.g., 0.1, 0.3, 0.6)", "0.2, 0.3, 0.5")
            params['weights'] = [float(x.strip()) for x in weights_str.split(",")]
            params['window'] = len(params['weights'])
        else:
            params['window'] = st.number_input("Lookback Window", 2, 24, 3)
            params['weights'] = None # Will calculate in logic

    elif technique == "Moving Average":
        params['window'] = st.slider("Moving Average Window", 2, 24, 3)

    elif technique == "Ramp Up Evenly":
        # New Requirement: Manually entering of Interval Ramp up Factor
        params['ramp_factor'] = st.number_input("Interval Ramp up Factor (Multiplier per period, e.g., 1.05 for 5% growth)", 
                                               min_value=0.0, value=1.05, step=0.01)

    elif technique == "Exponentially":
        params['alpha'] = st.slider("Smoothing Factor (Alpha)", 0.0, 1.0, 0.3)

# --- 3. Upload Data File ---
uploaded_file = st.file_uploader("Upload Data File (CSV/Excel)", type=['csv', 'xlsx'])

def calculate_steps(interval, horizon_label):
    days = horizon_map[horizon_label]
    if interval == "Daily": return days
    if interval == "Weekly": return max(1, days // 7)
    if interval == "Monthly": return max(1, days // 30)
    if interval == "Quarterly": return max(1, days // 90)
    if interval == "Year": return max(1, days // 365)
    return 24

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df['order_qty'] = pd.to_numeric(df['order_qty'], errors='coerce').fillna(0)

        # Handle Branching Logic (Aggregate vs Model vs Part)
        if main_option == "Aggregate Wise":
            working_df = df.copy()
            title_suffix = "Total Aggregate"
        else:
            if sub_option == "Model Wise":
                col_name = 'MODEL' if 'MODEL' in df.columns else 'PART DESCRIPTION'
                selection = st.selectbox(f"Select {sub_option}", df[col_name].unique())
                working_df = df[df[col_name] == selection].copy()
                title_suffix = f"Model: {selection}"
            else: # Part No Wise
                selection = st.selectbox("Select Part Number", df['PARTNO'].unique())
                working_df = df[df['PARTNO'] == selection].copy()
                title_suffix = f"Part: {selection}"

        # Resample data based on interval
        resample_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        working_df = working_df.set_index('Date').resample(resample_map[interval])['order_qty'].sum().reset_index()
        
        # --- 4. Generate Forecast & Trend Analysis ---
        history = working_df['order_qty'].values
        steps = calculate_steps(interval, horizon_label)
        forecast_values = []

        if len(history) > 0:
            if technique == "Historical Average":
                forecast_values = [np.mean(history)] * steps
                
            elif technique == "Moving Average":
                w = params['window']
                val = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
                forecast_values = [val] * steps
                
            elif technique == "Weightage Average":
                if params['weights']: # Manual
                    w = params['window']
                    if len(history) >= w:
                        val = np.dot(history[-w:], params['weights']) / sum(params['weights'])
                    else: val = np.mean(history)
                else: # Automated (Linear weights)
                    w = params['window']
                    if len(history) >= w:
                        weights = np.arange(1, w + 1)
                        val = np.dot(history[-w:], weights) / weights.sum()
                    else: val = np.mean(history)
                forecast_values = [val] * steps
                
            elif technique == "Ramp Up Evenly":
                base = history[-1] if len(history) > 0 else 0
                factor = params['ramp_factor']
                forecast_values = [base * (factor ** i) for i in range(1, steps + 1)]
                
            elif technique == "Exponentially":
                model = SimpleExpSmoothing(history).fit(smoothing_level=params['alpha'], optimized=False)
                forecast_values = model.forecast(steps)

            # Create Timeline
            last_date = working_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=resample_map[interval])[1:]

            # --- 5. Show Trend Analysis (Line Chart) ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=working_df['Date'], y=working_df['order_qty'], name="Past Actuals", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, name="Future Forecast", line=dict(color="#e67e22", dash='dot')))
            
            fig.update_layout(title=f"Trend Analysis: {title_suffix} ({technique})", 
                              xaxis_title="Timeline", yaxis_title="Quantity", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Results Display
            st.subheader("Forecasted Quantities")
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_values})
            st.dataframe(forecast_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}. Please ensure your file has 'Date', 'order_qty', 'PARTNO', and 'MODEL' columns.")
else:
    st.info("Follow the flowchart: Select your options, configure the technique, and upload your data file to see the Trend Analysis.")
