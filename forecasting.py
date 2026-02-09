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

st.title("📦 Intelligent Order Forecasting")

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

# --- STEP 3: UPLOAD DATA ---
st.markdown('<div class="step-header">STEP 3: Upload Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Wide-Format Excel/CSV", type=['xlsx', 'csv'])

# --- IMPROVED FLEXIBLE ENGINE ---
@st.cache_data(show_spinner=False)
def flexible_transform_data(file_content, is_csv):
    """Handles both DD-MM-YYYY and YYYY-MM-DD formats automatically"""
    raw = pd.read_csv(file_content) if is_csv else pd.read_excel(file_content)
    raw.columns = raw.columns.astype(str).str.strip()
    
    id_col = raw.columns[0]
    
    # Clean column headers: Filter only valid date headers
    valid_date_cols = []
    for col in raw.columns[1:]:
        # Try to convert header to date to see if it's a date column
        temp_date = pd.to_datetime(col, errors='coerce', dayfirst=True)
        if pd.notnull(temp_date):
            valid_date_cols.append(col)

    # Transform Wide to Long
    df_long = raw.melt(id_vars=[id_col], value_vars=valid_date_cols, var_name='RawDate', value_name='order_qty')
    
    # FIX: Use format='mixed' to handle different date strings in the same column
    df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce', format='mixed')
    
    # Ensure quantity is numeric
    df_long['order_qty'] = pd.to_numeric(df_long['order_qty'], errors='coerce').fillna(0)
    
    # Sort by date and remove any errors
    df_long = df_long.dropna(subset=['Date']).sort_values('Date')
    
    return df_long, id_col

def run_ai_forecast(data, horizon_days, interval_type):
    """XGBoost logic with improved date handling"""
    df = data.copy()
    if df.empty: return None, None

    # Feature engineering for XGBoost
    df['h'] = df['Date'].dt.hour
    df['d'] = df['Date'].dt.day
    df['m'] = df['Date'].dt.month
    df['y'] = df['Date'].dt.year
    df['dw'] = df['Date'].dt.dayofweek
    
    X = df[['h', 'd', 'm', 'y', 'dw']]
    y = df['order_qty']
    
    # Fast training
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # Future steps calculation
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    freq = res_map[interval_type]
    
    last_date = df['Date'].max()
    end_date = last_date + pd.Timedelta(days=horizon_days)
    
    future_dates = pd.date_range(start=last_date, end=end_date, freq=freq)
    if len(future_dates) <= 1:
        future_dates = pd.date_range(start=last_date, periods=2, freq=freq)
    
    future_dates = future_dates[1:] # Drop first date (which is already in history)

    f_df = pd.DataFrame({'Date': future_dates})
    f_df['h'] = f_df['Date'].dt.hour
    f_df['d'] = f_df['Date'].dt.day
    f_df['m'] = f_df['Date'].dt.month
    f_df['y'] = f_df['Date'].dt.year
    f_df['dw'] = f_df['Date'].dt.dayofweek
    
    preds = model.predict(f_df[['h', 'd', 'm', 'y', 'dw']])
    return future_dates, np.maximum(preds, 0)

# --- EXECUTION ---
if uploaded_file:
    try:
        df_long, id_col = flexible_transform_data(uploaded_file, uploaded_file.name.endswith('.csv'))
        
        # Filtering
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Aggregate Demand"
        else:
            options = df_long[id_col].unique()
            selected_item = st.selectbox(f"Select from {id_col}", options)
            target_df = df_long[df_long[id_col] == selected_item].copy()
            item_label = str(selected_item)

        # Resample to interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="step-header">STEP 4: Execution</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast and Trend Analysis", use_container_width=True):
            with st.spinner('AI analyzing history...'):
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                
                f_dates, f_values = run_ai_forecast(target_df, h_map[horizon_label], interval)

                if f_dates is not None:
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="History", line=dict(color="#2c3e50")))
                    fig.add_trace(go.Scatter(x=f_dates, y=f_values, name="AI Forecast", line=dict(color="#e67e22", dash='dot')))
                    fig.update_layout(title=f"Trend Analysis: {item_label}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    # Table
                    date_fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                    res_df = pd.DataFrame({"Forecast Date": f_dates.strftime(date_fmt), "Quantity": np.round(f_values, 1)})
                    st.dataframe(res_df, use_container_width=True)
                else:
                    st.error("No valid data found to forecast.")

    except Exception as e:
        st.error(f"Error processing data: {e}")
