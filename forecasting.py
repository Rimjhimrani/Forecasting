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

st.title("📦 Order Forecasting System")
st.info("Strict Mathematical Calculation Mode (Matches Excel Outputs).")

# --- FLOWCHART STEPS 1-3 ---
st.markdown('<div class="step-header">STEPS 1-3: Setup Options</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
    interval = st.selectbox("Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    sub_choice = st.radio("Select Level", ["Model Wise", "Part No Wise"], horizontal=True) if main_choice == "Product Wise" else None
    horizon_label = st.selectbox("Horizon", ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], index=2)

# --- FLOWCHART STEP 4: CHOOSE TECHNIQUE ---
st.markdown('<div class="step-header">STEP 4: Choose Forecast Techniques</div>', unsafe_allow_html=True)
technique = st.selectbox("Technique", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

# Technique specific inputs based on your formulas
tech_params = {}
if technique == "Weightage Average":
    w_mode = st.radio("Weight Mode", ["Automated", "Manual entering of weights"], horizontal=True)
    if w_mode == "Manual entering of weights":
        w_in = st.text_input("Enter weights (e.g., 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
        tech_params['weights'] = [float(x.strip()) for x in w_in.split(',')]
    else:
        tech_params['weights'] = [0.2, 0.3, 0.5]

elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Window Size (n)", 2, 30, 7)

elif technique == "Ramp Up Evenly":
    tech_params['n'] = st.number_input("Window Size for Ramp Up (n)", 2, 30, 7)
    # The PDF mentioned ramp factor, but your formula strictly uses 'n'. We will obey your formula strictly.

elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)

# --- YOUR EXACT FORMULAS ---
def historical_average(demand):
    if len(demand) == 0: return 0
    return sum(demand) / len(demand)

def weighted_average(demand, weights):
    n = len(weights)
    if len(demand) < n: return sum(demand) / len(demand) if demand else 0
    window = demand[-n:]
    return sum(d * w for d, w in zip(window, weights)) / sum(weights)

def moving_average(demand, n):
    if len(demand) == 0: return 0
    window = demand[-n:]
    return sum(window) / len(window)

def ramp_up_evenly(demand, n):
    if len(demand) == 0: return 0
    window = demand[-n:]
    actual_n = len(window)
    weights = list(range(1, actual_n + 1))
    total_weight = sum(weights)
    return sum(d * w for d, w in zip(window, weights)) / total_weight

def exponential_smoothing(demand, alpha):
    if len(demand) == 0: return 0
    forecast = demand[0]  # initial forecast
    for d in demand[1:]:
        forecast = alpha * d + (1 - alpha) * forecast
    return forecast

# --- STEP 5: UPLOAD DATA ---
st.markdown('<div class="step-header">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # Load and Transform Wide Format
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='order_qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
        df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter Logic
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_name = "Aggregate Total"
        else:
            options = df_long[id_col].unique()
            selected = st.selectbox(f"Select from {id_col}", options)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        # Resample Frequency
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()
        
        demand_history = target_df['order_qty'].tolist()

        # --- STEP 6: GENERATE ---
        st.markdown('<div class="step-header">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Analysis", use_container_width=True):
            if len(demand_history) == 0:
                st.error("No valid historical data found.")
            else:
                # 1. Calculate EXACT value using your formulas
                if technique == "Historical Average":
                    predicted_value = historical_average(demand_history)
                elif technique == "Weightage Average":
                    predicted_value = weighted_average(demand_history, tech_params['weights'])
                elif technique == "Moving Average":
                    predicted_value = moving_average(demand_history, tech_params['n'])
                elif technique == "Ramp Up Evenly":
                    predicted_value = ramp_up_evenly(demand_history, tech_params['n'])
                elif technique == "Exponentially":
                    predicted_value = exponential_smoothing(demand_history, tech_params['alpha'])

                # 2. Setup Future Dates
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
                
                if len(future_dates) == 0: 
                    future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                # Project the exact calculated formula forward
                preds = [predicted_value] * len(future_dates)

                # 3. Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Historical Data", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"Forecast ({technique})", line=dict(color="#e67e22", dash='dot')))
                fig.update_layout(title=f"Trend Analysis: {item_name}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. Data Table
                date_fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                res_df = pd.DataFrame({"Date": future_dates.strftime(date_fmt), "Predicted Qty": np.round(preds, 2)})
                st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
