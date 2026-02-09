import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# Set Page Config
st.set_page_config(page_title="Forecasting System", layout="centered")

# --- CUSTOM CSS FOR FLOWCHART FEEL ---
st.markdown("""
    <style>
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- APPLICATION START ---
st.title("📦 Order Forecasting & Trend System")
st.info("Follow the steps below to generate your forecast. Powered by XGBoost AI.")

# STEP 1: CHOOSE OPTION
st.markdown('<div class="step-header">STEP 1: Choose Option</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)

sub_choice = None
if main_choice == "Product Wise":
    sub_choice = st.radio("Product Level", ["Model Wise", "Part No Wise"], horizontal=True)

# STEP 2: SELECT INTERVAL & HORIZON
st.markdown('<div class="step-header">STEP 2: Select Timeline</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    interval = st.selectbox("Forecast Interval", 
                            ["Daily", "Weekly", "Monthly", "Quarterly", "Year"])

with col_b:
    horizon_label = st.selectbox("Forecast Horizon", 
                                ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"])

# STEP 3: UPLOAD
st.markdown('<div class="step-header">STEP 3: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'csv'])

# --- FIXED XGBOOST LOGIC (Hidden from user) ---
def run_xgb_forecast(data, steps, interval_type):
    df = data.copy()
    # Create simple features
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofweek'] = df['Date'].dt.dayofweek
    
    X = df[['day', 'month', 'year', 'dayofweek']]
    y = df['order_qty']
    
    # Locked optimized parameters for non-tech users
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # Predict future
    last_date = df['Date'].max()
    res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=res_map[interval_type])[1:]
    
    f_df = pd.DataFrame({'Date': future_dates})
    f_df['day'] = f_df['Date'].dt.day
    f_df['month'] = f_df['Date'].dt.month
    f_df['year'] = f_df['Date'].dt.year
    f_df['dayofweek'] = f_df['Date'].dt.dayofweek
    
    preds = model.predict(f_df[['day', 'month', 'year', 'dayofweek']])
    return future_dates, np.maximum(preds, 0)

# PROCESS DATA
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.str.strip()

        # Handle Wide Format (Image 1) or Long Format (Image 2)
        if 'MODEL' in raw.columns and any('-20' in str(c) for c in raw.columns):
            df_long = raw.melt(id_vars=['MODEL'], var_name='Date', value_name='order_qty')
            df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        else:
            df_long = raw.copy()
            df_long.rename(columns={'Part No': 'PARTNO', 'Qty/Veh': 'order_qty', 'Model': 'MODEL'}, inplace=True)
            df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')

        df_long = df_long.dropna(subset=['Date', 'order_qty'])

        # Dropdowns for specific item selection
        target_df = None
        item_label = "Total"
        
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
        elif sub_choice == "Model Wise":
            m_list = df_long['MODEL'].unique()
            selected_m = st.selectbox("Select Model to Forecast", m_list)
            target_df = df_long[df_long['MODEL'] == selected_m].groupby('Date')['order_qty'].sum().reset_index()
            item_label = selected_m
        else:
            p_list = df_long['PARTNO'].unique()
            selected_p = st.selectbox("Select Part Number to Forecast", p_list)
            target_df = df_long[df_long['PARTNO'] == selected_p].groupby('Date')['order_qty'].sum().reset_index()
            item_label = selected_p

        # STEP 4: GENERATE
        st.markdown('<div class="step-header">STEP 4: Execution</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast and Trend Analysis", use_container_width=True):
            
            # Map Horizon to steps
            h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 365*3, "5 years": 365*5}
            steps = h_map[horizon_label]
            if interval == "Weekly": steps //= 7
            if interval == "Monthly": steps //= 30
            if interval == "Quarterly": steps //= 90
            if interval == "Year": steps //= 365
            steps = max(1, int(steps))

            # Run ML
            f_dates, f_values = run_xgb_forecast(target_df, steps, interval)

            # PLOT
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="Past Data", line=dict(color="#2c3e50")))
            fig.add_trace(go.Scatter(x=f_dates, y=f_values, name="AI Forecast", line=dict(color="#e74c3c", dash='dot')))
            fig.update_layout(title=f"Trend Analysis for {item_label}", template="plotly_white")
            st.plotly_chart(fig)

            # DISPLAY TABLE
            st.subheader("Future Forecast Values")
            res_df = pd.DataFrame({"Date": f_dates.strftime('%d-%m-%Y'), "Predicted Qty": np.round(f_values, 1)})
            st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file. Please check your columns. Error: {e}")
