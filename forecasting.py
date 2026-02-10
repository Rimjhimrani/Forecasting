import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# --- UI SETTINGS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf, #2e7bcf); color: white; }
    div.stButton > button:first-child {
        background-color: #007bff; color: white; width: 100%; border-radius: 5px; height: 3em; font-weight: bold;
    }
    .status-box { padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; background-color: white; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION (Steps 1-4) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1659/1659066.png", width=80)
    st.title("Forecast Settings")
    
    st.subheader("Step 1: Scope")
    main_choice = st.radio("Primary Path", ["Aggregate Wise", "Product Wise"])
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Product Level", ["Model Wise", "Part No Wise"])

    st.subheader("Step 2: Time Frequency")
    interval = st.select_slider("Select Frequency", 
                                options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], 
                                value="Daily")

    st.subheader("Step 3: Horizon")
    horizon_label = st.selectbox("Future Length", 
                                ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"], 
                                index=2)

    st.subheader("Step 4: AI Strategy")
    technique = st.selectbox("Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
    
    tech_params = {}
    if technique == "Weightage Average":
        w_mode = st.radio("Weight Selection", ["Automated", "Manual"])
        if w_mode == "Manual":
            w_in = st.text_input("Enter weights (e.g. 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
            tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        else: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05, 0.01)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)

# --- BASELINE LOGIC ---
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
        d_slice = demand[-7:]
        weights = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, weights) / weights.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

# --- MAIN PAGE ---
st.title("📊 AI Precision Supply Chain Forecast")
st.markdown("Predict demand using a hybrid of traditional statistical baselines and XGBoost AI.")

# Step 5: Upload
upload_container = st.container()
with upload_container:
    uploaded_file = st.file_uploader("Upload your Supply Chain Data (Excel/CSV with dates as columns)", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # 1. Process Data
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        # Filter and Aggregation
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate Total"
        else:
            selected = st.selectbox(f"Select Specific {sub_choice}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # UI Layout for Results
        st.divider()
        col_btn, col_info = st.columns([1, 3])
        
        with col_btn:
            execute = st.button("🚀 RUN PREDICTION")
        
        if execute:
            with st.spinner('Applying Hybrid AI logic...'):
                history = target_df['qty'].tolist()
                excel_base = calculate_excel_baseline(history, technique, tech_params)
                
                # AI Model Logic
                target_df['month'] = target_df['Date'].dt.month
                target_df['dow'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - excel_base
                
                model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
                model.fit(target_df[['month', 'dow']], target_df['diff'])
                
                # Future Generation
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
                
                if len(future_dates) == 0: 
                    future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                f_df = pd.DataFrame({'Date': future_dates})
                f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                ai_wiggles = model.predict(f_df[['month', 'dow']])
                
                final_preds = np.maximum(excel_base + ai_wiggles, 0)
                if technique == "Ramp Up Evenly":
                    final_preds = [p * (tech_params['ramp_factor'] ** i) for i, p in enumerate(final_preds, 1)]

                # --- DASHBOARD DISPLAY ---
                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Item", item_name)
                m2.metric("Baseline Qty", f"{excel_base:.2f}")
                m3.metric("Forecast Peak", f"{np.max(final_preds):.2f}")

                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], 
                                         name="Historical Demand", line=dict(color="#2c3e50", width=2)))
                fig.add_trace(go.Scatter(x=future_dates, y=final_preds, 
                                         name="AI Forecast", line=dict(color="#FF8C00", width=3, dash='dot')))
                
                fig.update_layout(
                    title=f"Forecast Trend: {item_name}",
                    hovermode="x unified",
                    plot_bgcolor="white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                
                st.plotly_chart(fig, use_container_width=True)

                # Data Table
                with st.expander("View Prediction Table"):
                    date_fmt = '%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'
                    res_df = pd.DataFrame({
                        "Forecast Date": future_dates.strftime(date_fmt), 
                        "Predicted Quantity": np.round(final_preds, 2)
                    })
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("👋 Welcome! Please upload your data file in the sidebar or main area to begin.")
