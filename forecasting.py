import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Order Forecast", layout="wide", initial_sidebar_state="expanded")

# --- MODERN STYLING ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #4338CA; border: none; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
    }
    .header-text { color: #1E293B; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1681/1681283.png", width=80)
    st.markdown("# Settings")
    
    st.markdown("### 1. Aggregation")
    agg_mode = st.radio("Level", ["Aggregate Wise", "Product Wise"], label_visibility="collapsed")
    
    st.markdown("### 2. Time Control")
    interval = st.selectbox("Frequency", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    
    st.markdown("### 3. Forecasting Logic")
    technique = st.selectbox("Formula", ["Historical Average", "Moving Average", "Weightage Average", "Ramp Up Evenly", "Exponentially"])
    
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Weights", "0.2, 0.3, 0.5")
        tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
    elif technique == "Moving Average":
        tech_params['n'] = st.slider("Window Size", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Alpha (Smoothing)", 0.01, 1.0, 0.3)

# --- MAIN CONTENT ---
st.markdown('<h1 class="header-text">📦 AI Order Forecasting</h1>', unsafe_allow_html=True)
st.markdown("Upload your historical data to generate AI-powered supply chain predictions.")

# Step 1: Upload
upload_col, action_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['xlsx', 'csv'])

with action_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_file:
        generate_btn = st.button("🚀 Run Analysis")
    else:
        st.info("Please upload a file to begin.")
        generate_btn = False

# Logic: Data Processing
def calculate_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', [0.3, 0.7])
        n = len(w)
        return np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
    elif tech == "Ramp Up Evenly":
        return np.mean(demand[-7:]) if len(demand) >= 7 else np.mean(demand)
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

if uploaded_file:
    try:
        # Load Data
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        id_col = df_raw.columns[0]
        date_cols = [c for c in df_raw.columns[1:] if pd.to_datetime(c, errors='coerce') is not pd.NaT]
        
        # Melt and Clean
        df_long = df_raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='Date', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Filter Logic
        if agg_mode == "Product Wise":
            product = st.selectbox("Select Product/Model", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == product].copy()
            title_suffix = f"for {product}"
        else:
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            title_suffix = "(Total Aggregated)"

        # Resample
        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if generate_btn:
            with st.spinner('AI is analyzing patterns...'):
                # 1. Calculation
                history = target_df['qty'].tolist()
                base_val = calculate_baseline(history, technique, tech_params)
                
                # AI Model (XGBoost)
                target_df['month'] = target_df['Date'].dt.month
                target_df['dow'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - base_val
                
                model = XGBRegressor(n_estimators=50)
                model.fit(target_df[['month', 'dow']], target_df['diff'])
                
                # Future Dates
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, periods=13, freq=res_map[interval])[1:]
                
                # Predictions
                f_df = pd.DataFrame({'Date': future_dates})
                f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                ai_resid = model.predict(f_df[['month', 'dow']])
                
                excel_fc = []
                ai_fc = []
                for i, res in enumerate(ai_resid, 1):
                    mult = (tech_params.get('ramp_factor', 1.05)**i) if technique == "Ramp Up Evenly" else 1
                    b = base_val * mult
                    excel_fc.append(round(b, 2))
                    ai_fc.append(round(max(b + res, 0), 2))

                # --- VISUALS ---
                tab1, tab2 = st.tabs(["📊 Forecast Chart", "📋 Data Table"])
                
                with tab1:
                    fig = go.Figure()
                    # Historical
                    fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Actual Demand", line=dict(color='#6366F1', width=3)))
                    # Excel
                    fig.add_trace(go.Scatter(x=future_dates, y=excel_fc, name="Baseline Formula", line=dict(color='#94A3B8', dash='dot')))
                    # AI
                    fig.add_trace(go.Scatter(x=future_dates, y=ai_fc, name="AI Prediction", line=dict(color='#10B981', width=4)))
                    
                    fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Wiggle Bar Chart
                    st.markdown("### 🧠 AI Pattern Adjustments")
                    st.caption("How much the AI adds/subtracts from the baseline based on seasonality.")
                    fig_bar = go.Figure(go.Bar(x=future_dates, y=ai_resid, marker_color='#6366F1'))
                    fig_bar.update_layout(template="plotly_white", height=250)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with tab2:
                    out_df = pd.DataFrame({
                        "Date": future_dates.strftime('%Y-%m-%d'),
                        "AI Forecast": ai_fc,
                        "Baseline": excel_fc
                    })
                    st.dataframe(out_df, use_container_width=True)
                    
                    # Download
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        out_df.to_excel(writer, index=False)
                    st.download_button("📥 Download Forecast Results", data=output.getvalue(), file_name="AI_Forecast.xlsx")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
