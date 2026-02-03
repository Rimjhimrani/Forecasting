import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from datetime import timedelta

st.set_page_config(page_title="Order Forecasting System", layout="wide")

# --- UI Header ---
st.title("📊 Order Forecasting System")
st.markdown("---")

# --- STEP 1: Choose an Option (Flowchart Step 3) ---
col1, col2 = st.columns(2)
with col1:
    analysis_type = st.radio("Choose Analysis Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

# --- STEP 2: Select Forecast Interval & Horizon (Flowchart Step 4 & 5) ---
with st.sidebar:
    st.header("Settings")
    
    interval = st.selectbox("Select Forecast Interval", 
                            ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"])
    
    horizon_map = {
        "Day": 1, "Week": 7, "Month": 30, "Quarter": 90, 
        "Year": 365, "3 years": 365*3, "5 years": 365*5
    }
    horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_map.keys()))

    # --- STEP 3: Choose Forecast Techniques (Flowchart Step 6 & 7) ---
    st.subheader("Forecast Techniques")
    technique = st.selectbox("Method", 
                             ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
    
    # Parameters for specific techniques
    params = {}
    if technique == "Moving Average" or technique == "Weightage Average":
        params['window'] = st.slider("Window Size (Periods)", 2, 24, 3)
    if technique == "Ramp Up Evenly":
        params['growth'] = st.slider("Growth Rate (%)", 0, 50, 5) / 100

# --- STEP 4: Upload Data File (Flowchart Step 8) ---
uploaded_file = st.file_uploader("Upload your Data File (CSV or Excel)", type=['csv', 'xlsx'])

# Helper function to calculate steps needed for horizon
def calculate_steps(interval, horizon_label):
    # Rough estimate of steps based on selected interval and horizon
    days = horizon_map[horizon_label]
    if interval == "Daily": return days
    if interval == "Weekly": return max(1, days // 7)
    if interval == "Monthly": return max(1, days // 30)
    if interval == "Quarterly": return max(1, days // 90)
    if interval == "Year": return max(1, days // 365)
    return 24 # Default for hourly

if uploaded_file:
    # Read Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Standardize Columns
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df['order_qty'] = pd.to_numeric(df['order_qty'], errors='coerce').fillna(0)
        
        # --- Logic for Product Wise vs Aggregate Wise ---
        if analysis_type == "Product Wise":
            parts = df['PARTNO'].unique()
            selected_part = st.selectbox("Select Product (PARTNO)", parts)
            working_df = df[df['PARTNO'] == selected_part].copy()
        else:
            st.info("Aggregating data for all products (Total Store View)")
            working_df = df.copy()

        # --- Resampling based on Interval ---
        resample_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        working_df = working_df.set_index('Date').resample(resample_map[interval])['order_qty'].sum().reset_index()
        
        # --- STEP 5 & 6: Generate Forecast & Trend Analysis (Flowchart Step 9 & 10) ---
        history = working_df['order_qty'].values
        steps = calculate_steps(interval, horizon_label)
        forecast_values = []

        if len(history) > 0:
            if technique == "Historical Average":
                val = np.mean(history)
                forecast_values = [val] * steps
                
            elif technique == "Moving Average":
                w = params['window']
                val = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
                forecast_values = [val] * steps
                
            elif technique == "Weightage Average":
                w = params['window']
                if len(history) >= w:
                    weights = np.arange(1, w + 1)
                    val = np.dot(history[-w:], weights) / weights.sum()
                else: val = np.mean(history)
                forecast_values = [val] * steps
                
            elif technique == "Ramp Up Evenly":
                base = np.mean(history[-3:]) if len(history) >= 3 else np.mean(history)
                forecast_values = [base * (1 + params['growth'] * i) for i in range(1, steps + 1)]
                
            elif technique == "Exponentially":
                try:
                    model = SimpleExpSmoothing(history, initialization_method="estimated").fit()
                    forecast_values = model.forecast(steps)
                except:
                    forecast_values = [history[-1]] * steps

            # Create Future Dates
            last_date = working_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=resample_map[interval])[1:]

            # --- Visualization ---
            fig = go.Figure()
            # Past Data
            fig.add_trace(go.Scatter(x=working_df['Date'], y=working_df['order_qty'], 
                                     name="Historical Data", line=dict(color="#1f77b4", width=2)))
            # Forecast Data
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, 
                                     name=f"Forecast ({technique})", line=dict(color="#ff7f0e", dash='dash')))
            
            fig.update_layout(title=f"{analysis_type} Trend Analysis - {interval} Interval",
                              xaxis_title="Timeline", yaxis_title="Quantity",
                              template="plotly_white", hovermode="x unified")
            
            st.plotly_chart(fig, use_container_width=True)

            # --- Data Table ---
            st.subheader("Forecasted Figures")
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Qty": forecast_values})
            st.dataframe(forecast_df, use_container_width=True)
            
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "forecast_report.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}. Please ensure your file has 'Date', 'order_qty', and 'PARTNO' columns.")

else:
    st.info("Please upload a data file to begin the analysis.")
