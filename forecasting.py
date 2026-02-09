import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# --- UI SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="Forecasting Flow System", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .step-box {
        background-color: #00B0F0;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        font-size: 20px;
        border: 2px solid #000;
    }
    .start-box {
        background-color: #FFFF00;
        color: black;
        padding: 10px;
        border: 2px solid #000;
        text-align: center;
        font-weight: bold;
        width: 150px;
        margin: auto;
    }
    .end-box {
        background-color: #00B050;
        color: white;
        padding: 10px;
        border: 2px solid #000;
        text-align: center;
        font-weight: bold;
        width: 150px;
        margin: auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #ffffff;
        border: 2px solid #000;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- START OF PROCESS ---
st.markdown('<div class="start-box">START</div>', unsafe_allow_html=True)
st.title("📊 Supply Chain Forecasting System")
st.divider()

# --- FLOWCHART STEP 1: CHOOSE AN OPTION ---
st.markdown('<div class="step-box">STEP 1: Choose an Option</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_option = st.radio("Primary Selection", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    if main_option == "Product Wise":
        sub_option = st.radio("Product Level", ["Model Wise", "Part No Wise"], horizontal=True)
    else:
        st.write("Full data aggregation selected.")
        sub_option = None

# --- FLOWCHART STEP 2: SELECT INTERVAL ---
st.markdown('<div class="step-box">STEP 2: Select Forecast Interval</div>', unsafe_allow_html=True)
interval = st.select_slider(
    "Select Granularity",
    options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"],
    value="Daily"
)

# --- FLOWCHART STEP 3: SELECT HORIZON ---
st.markdown('<div class="step-box">STEP 3: Select Forecast Horizon</div>', unsafe_allow_html=True)
horizon_label = st.selectbox(
    "How far into the future?",
    ["Day", "Week", "Month", "Quarter", "Year", "3 years", "5 years"],
    index=2
)

# --- FLOWCHART STEP 4: CHOOSE TECHNIQUES ---
st.markdown('<div class="step-box">STEP 4: Choose Forecast Technique</div>', unsafe_allow_html=True)
technique = st.selectbox("Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

# Technique Specific Inputs (From Flowchart Boxes)
t_col1, t_col2 = st.columns([2, 1])
tech_params = {}
with t_col1:
    if technique == "Weightage Average":
        st.info("💡 Formula: Even Weights will be calculated by automated OR enter manual.")
        w_mode = st.radio("Weighting Mode", ["Automated (Even)", "Manual entry"], horizontal=True)
        if w_mode == "Manual entry":
            w_in = st.text_input("Enter weights (e.g. 0.2, 0.3, 0.5)", "0.2, 0.3, 0.5")
            tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        else:
            tech_params['weights'] = np.array([0.33, 0.33, 0.34])
            
    elif technique == "Ramp Up Evenly":
        st.info("💡 Formula: Applied growth factor over AI baseline.")
        tech_params['ramp_factor'] = st.number_input("Manually entering of Interval Ramp up Factor", 1.0, 2.0, 1.05, 0.01)

    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
    
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)

# --- FLOWCHART STEP 5: UPLOAD DATA FILE ---
st.markdown('<div class="step-box">STEP 5: Upload the Data File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop your Wide-Format Excel/CSV here", type=['xlsx', 'csv'])

# --- HYBRID ENGINE LOGIC ---
def get_excel_math(demand, tech, params):
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average": return np.mean(demand[-params.get('n', 7):])
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.33, 0.33, 0.34]))
        n = len(w)
        return np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
    elif tech == "Ramp Up Evenly":
        d_slice = demand[-7:]
        w = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, w) / w.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        f = demand[0]
        for d in demand[1:]: f = alpha * d + (1 - alpha) * f
        return f
    return np.mean(demand)

# --- FLOWCHART STEP 6: GENERATE ---
if uploaded_file:
    try:
        # Data Transformation (Wide to Long)
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        # Filtering
        if main_option == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate"
        else:
            selected = st.selectbox(f"Select {sub_option}", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="step-box" style="background-color:#FFFF00; color:black;">STEP 6: Generate Forecast and Trend</div>', unsafe_allow_html=True)
        if st.button("🚀 CLICK HERE TO GENERATE ANALYSIS"):
            with st.spinner('AI analyzing your data flow...'):
                history = target_df['qty'].tolist()
                
                # Math Baseline
                base = get_excel_math(history, technique, tech_params)
                
                # XGBoost Pattern
                target_df['m'] = target_df['Date'].dt.month
                target_df['d'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - np.mean(history)
                
                model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
                model.fit(target_df[['m', 'd']], target_df['diff'])
                
                # Horizon calc
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365, "3 years": 1095, "5 years": 1825}
                last_date = target_df['Date'].max()
                future_dates = pd.date_range(start=last_date, end=last_date + pd.Timedelta(days=h_map[horizon_label]), freq=res_map[interval])[1:]
                
                if len(future_dates) == 0: future_dates = pd.date_range(start=last_date, periods=2, freq=res_map[interval])[1:]

                f_df = pd.DataFrame({'Date': future_dates})
                f_df['m'], f_df['d'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                
                ai_wiggle = model.predict(f_df[['m', 'd']])
                final_preds = base + ai_wiggle
                
                if technique == "Ramp Up Evenly":
                    final_preds = [p * (tech_params['ramp_factor'] ** i) for i, p in enumerate(final_preds, 1)]

                # Trend Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Historical Data", line=dict(color="#2c3e50", width=2)))
                fig.add_trace(go.Scatter(x=future_dates, y=final_preds, name="AI Future Forecast", line=dict(color="#FF8C00", width=3, dash='dot')))
                fig.update_layout(title=f"Show Trend Analysis (Line Chart) for {item_name}", template="plotly_white", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Results Table
                st.subheader("📋 Forecasted Data Table")
                res_df = pd.DataFrame({
                    "Date": future_dates.strftime('%d-%m-%Y %H:%M' if interval=="Hourly" else '%d-%m-%Y'), 
                    "Quantity": np.round(final_preds, 1)
                })
                st.dataframe(res_df, use_container_width=True)
                
                st.markdown('<div class="end-box">END</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in data flow: {e}")
else:
    st.warning("Please upload the Data File at Step 5 to continue the process.")
