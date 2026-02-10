import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import timedelta

# --- UI SETTINGS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .step-header {
        color: #1c2e4a;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #007bff;
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1659/1659104.png", width=80)
    st.title("Settings")
    
    st.markdown('<div class="step-header">1. Scope</div>', unsafe_allow_html=True)
    main_choice = st.radio("Primary Path", ["Aggregate Wise", "Product Wise"])
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"])

    st.markdown('<div class="step-header">2. Frequency</div>', unsafe_allow_html=True)
    interval = st.selectbox("Data Interval", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"])

    st.markdown('<div class="step-header">3. Forecast Horizon</div>', unsafe_allow_html=True)
    h_unit = st.selectbox("Predict for the next:", ["Days", "Weeks", "Months", "Years"])
    h_value = st.number_input(f"Number of {h_unit}", min_value=1, value=6)

    st.markdown('<div class="step-header">4. AI Strategy</div>', unsafe_allow_html=True)
    technique = st.selectbox("Technique", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

    tech_params = {}
    if technique == "Weightage Average":
        w_mode = st.radio("Weights", ["Automated", "Manual"])
        if w_mode == "Manual":
            w_in = st.text_input("Values (comma separated)", "0.2, 0.3, 0.5")
            tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        else: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.slider("Window Size", 2, 30, 7)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing (Alpha)", 0.01, 1.0, 0.3)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)

# --- MAIN CONTENT ---
st.title("📊 Supply Chain AI Forecaster")
st.info("Upload your historical data to generate high-precision AI forecasts.")

# --- STEP 5: UPLOAD ---
st.markdown('<div class="step-header">Upload Data</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Excel/CSV (Dates as columns)", type=['xlsx', 'csv'])

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

if uploaded_file:
    try:
        # Data Processing
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.to_datetime(c, errors='coerce', dayfirst=True) is not None]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        # Filtering Selection
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Total Aggregate"
        else:
            options = df_long[id_col].unique()
            selected = st.selectbox(f"Search & Select {sub_choice}", options)
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # Forecasting Logic
        st.markdown("---")
        if st.button("🚀 EXECUTE AI FORECAST"):
            with st.spinner('Calculating Hybrid AI Trend...'):
                history = target_df['qty'].tolist()
                excel_base = calculate_excel_baseline(history, technique, tech_params)
                
                # AI Model Training
                target_df['month'] = target_df['Date'].dt.month
                target_df['dow'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - excel_base
                
                model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
                model.fit(target_df[['month', 'dow']], target_df['diff'])
                
                # Calculate Future Range based on user input
                last_date = target_df['Date'].max()
                delta_map = {"Days": "days", "Weeks": "weeks", "Months": "days", "Years": "days"}
                
                # Approximation for months/years
                if h_unit == "Months": offset = timedelta(days=h_value * 30)
                elif h_unit == "Years": offset = timedelta(days=h_value * 365)
                elif h_unit == "Weeks": offset = timedelta(weeks=h_value)
                else: offset = timedelta(days=h_value)

                future_dates = pd.date_range(start=last_date, end=last_date + offset, freq=res_map[interval])[1:]
                if len(future_dates) == 0: 
                    future_dates = pd.date_range(start=last_date, periods=h_value + 1, freq=res_map[interval])[1:]

                f_df = pd.DataFrame({'Date': future_dates})
                f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                ai_wiggles = model.predict(f_df[['month', 'dow']])
                
                final_preds = excel_base + ai_wiggles
                final_preds = np.maximum(final_preds, 0)
                
                if technique == "Ramp Up Evenly":
                    final_preds = [p * (tech_params['ramp_factor'] ** i) for i, p in enumerate(final_preds, 1)]

                # Visualization
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="History", line=dict(color="#1c2e4a", width=2)))
                    fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name="AI Forecast", line=dict(color="#FF8C00", dash='dot', width=3)))
                    fig.update_layout(
                        title=f"Forecast Analysis: {item_name}",
                        hovermode="x unified",
                        template="plotly_white",
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.markdown("**Forecasting Metrics**")
                    st.metric("Baseline Value", f"{excel_base:.2f}")
                    st.metric("Forecast Peak", f"{np.max(final_preds):.2f}")
                    st.metric("Future Points", len(future_dates))

                # Results Table
                st.markdown('<div class="step-header">Data Table</div>', unsafe_allow_html=True)
                date_fmt = '%d-%m-%Y' if interval != "Hourly" else '%d-%m-%Y %H:%M'
                res_df = pd.DataFrame({
                    "Date": future_dates.strftime(date_fmt), 
                    "Forecasted Qty": np.round(final_preds, 2)
                })
                st.dataframe(res_df, use_container_width=True, height=300)

                # Export
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    # Placeholder UI
    st.image("https://cdn.dribbble.com/users/1489147/screenshots/5732100/data_illustration_4x.png", width=400)
    st.warning("Please upload a file in the main section to begin.")
