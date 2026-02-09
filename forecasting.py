import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI Setup ---
st.set_page_config(page_title="Fast Forecasting System", layout="centered")

st.markdown("""
    <style>
    .step-header {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin-bottom: 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Fast AI Order Forecasting")

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

# --- SPEED OPTIMIZED AI ENGINE ---
@st.cache_data(show_spinner=False)
def fast_transform_data(file_content, is_csv):
    """Caches the heavy data transformation process"""
    raw = pd.read_csv(file_content) if is_csv else pd.read_excel(file_content)
    raw.columns = raw.columns.str.strip()
    id_col = raw.columns[0]
    
    # Melt wide to long
    df_long = raw.melt(id_vars=[id_col], var_name='Date', value_name='order_qty')
    df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
    df_long = df_long.dropna(subset=['Date'])
    return df_long, id_col

def run_fast_forecast(data, horizon_days, interval_type):
    """Optimized XGBoost Training"""
    df = data.copy()
    
    # Fast Feature Extraction
    df['h'] = df['Date'].dt.hour
    df['d'] = df['Date'].dt.day
    df['m'] = df['Date'].dt.month
    df['y'] = df['Date'].dt.year
    df['dw'] = df['Date'].dt.dayofweek
    
    X = df[['h', 'd', 'm', 'y', 'dw']]
    y = df['order_qty']
    
    # Faster parameters: Reduced n_estimators from 500 to 100
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Forecast Generation
    res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
    freq = res_map[interval_type]
    
    last_date = df['Date'].max()
    end_date = last_date + pd.Timedelta(days=horizon_days)
    
    future_dates = pd.date_range(start=last_date, end=end_date, freq=freq)[1:]
    if len(future_dates) == 0:
        future_dates = pd.date_range(start=last_date, periods=2, freq=freq)[1:]

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
        # Load and transform (cached)
        df_long, id_col = fast_transform_data(uploaded_file, uploaded_file.name.endswith('.csv'))
        
        # Filter Logic
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['order_qty'].sum().reset_index()
            item_label = "Total Aggregate"
        else:
            options = df_long[id_col].unique()
            selected_item = st.selectbox(f"Select {id_col}", options)
            target_df = df_long[df_long[id_col] == selected_item].copy()
            item_label = str(selected_item)

        # Consolidate interval
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="step-header">STEP 4: Execution</div>', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast Now", use_container_width=True):
            with st.spinner('AI is calculating...'):
                h_days = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                
                # Run the forecast
                f_dates, f_values = run_fast_forecast(target_df, h_days[horizon_label], interval)

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['order_qty'], name="History", line=dict(color="#2c3e50")))
                fig.add_trace(go.Scatter(x=f_dates, y=f_values, name="Forecast", line=dict(color="#28a745", dash='dot')))
                fig.update_layout(title=f"Trend Analysis: {item_label}", template="plotly_white", margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

                # Data Table
                fmt = '%d-%m-%Y %H:%M' if interval == "Hourly" else '%d-%m-%Y'
                res_df = pd.DataFrame({"Date": f_dates.strftime(fmt), "Forecast Qty": np.round(f_values, 1)})
                st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
