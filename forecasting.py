import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. SETTINGS & LUXURY DARK UI CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    /* Main Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* Custom Titles */
    .main-title {
        font-size: 40px;
        font-weight: 600;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    /* Step Headers in Sidebar */
    .sidebar-step {
        color: #38bdf8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 20px;
        margin-bottom: 10px;
        display: block;
    }

    /* Glowing Execute Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
    }

    /* Input focus colors */
    input, select {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: STEPS 1, 2, & 3 ---
with st.sidebar:
    st.markdown("<h2 style='color: white;'>Settings</h2>", unsafe_allow_html=True)
    
    # Step 1: Scope
    st.markdown('<span class="sidebar-step">Step 1: Scope Selection</span>', unsafe_allow_html=True)
    main_choice = st.radio("Selection Path", ["Aggregate Wise", "Product Wise"])
    if main_choice == "Product Wise":
        sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"])
    
    # Step 2: Timeline
    st.markdown('<span class="sidebar-step">Step 2: Timeline</span>', unsafe_allow_html=True)
    interval = st.selectbox("Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    horizon_label = st.selectbox("Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
    
    # Step 3: Technique
    st.markdown('<span class="sidebar-step">Step 3: Strategy</span>', unsafe_allow_html=True)
    technique = st.selectbox("Baseline Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
    
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Weights (eg: 0.2, 0.8)", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Window", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Multiplier", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Alpha", 0.01, 1.0, 0.3)

# --- MAIN CONTENT AREA ---
st.markdown('<h1 class="main-title">AI Precision Forecast</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; margin-bottom: 30px;">Advanced Supply Chain Intelligence Panel</p>', unsafe_allow_html=True)

# Step 4: Data Ingestion (In a Glass Card)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<span style="color: #38bdf8; font-weight: 600;">STEP 4: DATA INGESTION</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop your historical CSV or Excel file here", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE CALCULATION LOGIC (Preserved) ---
def calculate_excel_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.5, 0.5]))
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

# --- EXECUTION FLOW ---
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        id_col = raw.columns[0]
        date_cols = [c for c in raw.columns[1:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        
        df_long = raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Aggregate Total"
        else:
            selected = st.selectbox("🎯 Target Identification", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.write("")
        if st.button("RUN PREDICTIVE ENGINE"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            
            # --- RESULTS SECTION ---
            col_graph, col_stats = st.columns([3, 1])
            
            with col_stats:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🛠 Adjust View")
                dynamic_val = st.number_input("Lookahead Value", min_value=1, value=12)
                dynamic_unit = st.selectbox("Unit", ["Days", "Weeks", "Months", "Original Selection"])
                st.markdown('</div>', unsafe_allow_html=True)

            # Calculations
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base_scalar
            
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            last_date, last_qty = target_df['Date'].max(), target_df['qty'].iloc[-1]
            
            if dynamic_unit == "Original Selection":
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
                end_date = last_date + pd.Timedelta(days=h_map[horizon_label])
            elif dynamic_unit == "Days": end_date = last_date + pd.Timedelta(days=dynamic_val)
            elif dynamic_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=dynamic_val)
            else: end_date = last_date + pd.DateOffset(months=dynamic_val)
            
            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            excel_calc_col, predicted_calc_col = [], []
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- PLOTTING ---
            with col_graph:
                fig = go.Figure()
                # History
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Historical", line=dict(color="#6366f1", width=3, shape='spline')))
                
                f_dates_conn = [last_date] + list(future_dates)
                # Baseline
                fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty] + excel_calc_col, name="Excel Baseline", line=dict(color="rgba(255,255,255,0.3)", width=2, dash='dot')))
                # AI Forecast
                fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty] + predicted_calc_col, name="AI Forecast", line=dict(color="#38bdf8", width=4, shape='spline')))
                
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified", height=450, margin=dict(l=0,r=0,t=20,b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- AI WIGGLES ---
            st.markdown("### 🌊 AI Seasonality Corrections")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, marker_color="#818cf8"))
            fig_wig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=200, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- TABLE & DOWNLOAD ---
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            res_df = pd.DataFrame({"Date": future_dates.strftime('%Y-%m-%d'), "Predicted AI": predicted_calc_col, "Excel Baseline": excel_calc_col})
            st.dataframe(res_df, use_container_width=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("Download Prediction Report", output.getvalue(), f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Engine Error: {e}")
