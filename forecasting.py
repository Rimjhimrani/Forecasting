import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

if "wa_weights" not in st.session_state:
    st.session_state.wa_weights = None

# --- 1. PREMIUM ENTERPRISE UI CONFIG ---
st.set_page_config(page_title="AI Supply Chain | Precision", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #111827;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 5rem;
        max-width: 900px;
        margin: 0 auto;
    }

    .step-wrapper {
        position: relative;
        padding-left: 40px;
        margin-bottom: 50px;
        border-left: 2px solid #E5E7EB;
    }

    .step-dot {
        position: absolute;
        left: -9px;
        top: 0;
        width: 16px;
        height: 16px;
        background-color: #FFFFFF;
        border: 2px solid #4F46E5;
        border-radius: 50%;
    }

    .step-label {
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }

    .step-heading {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 20px;
    }

    .stSelectbox, .stRadio, .stNumberInput, .stTextInput {
        background-color: #F9FAFB;
        border-radius: 8px;
    }

    div.stButton > button {
        width: 100%;
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 18px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #4F46E5 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .insight-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 30px;
        border-radius: 20px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div style="text-align: center; margin-bottom: 80px;">'
            '<h1 style="font-size: 3rem; font-weight: 800; letter-spacing: -2px; color: #4F46E5;">Agilo<span style="color:#111827;">Forecast</span></h1>'
            '<p style="color: #6B7280; font-size: 1.2rem;">AI-Driven Intelligence for Forecasting & Decisions</p>'
            '</div>', unsafe_allow_html=True)

# --- STEP 1 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 01</div><div class="step-heading">Forecasting Scope</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Resolution Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 02</div><div class="step-heading">Set Parameters</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    horizon_label = st.selectbox("Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 03</div><div class="step-heading">Forecast Techniques</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

unit_map = {"Hourly": "Hours", "Daily": "Days", "Weekly": "Weeks", "Monthly": "Months", "Quarterly": "Quarters", "Year": "Years"}
current_unit = unit_map.get(interval, "Periods")

with c3:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c4:
    tech_params = {}
    if technique == "Weightage Average":
        w_mode = st.radio("Weight Configuration", ["Manual Entry", "Automated (Evenly)"], horizontal=True, key="wa_mode")
        if w_mode == "Manual Entry":
            w_in = st.text_input("Manual Ratios (comma separated)", value="0.3,0.7", key="wa_input")
            try: weights = np.array([float(x.strip()) for x in w_in.split(",")])
            except: weights = np.array([0.5, 0.5])
        else:
            w_lookback = st.number_input(f"Lookback ({current_unit})", 1, 100, 3, key="wa_lookback")
            weights = np.ones(w_lookback) / w_lookback
        tech_params["weights"] = weights
            
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input(f"Lookback Window ({current_unit})", min_value=1, max_value=100, value=7)
    
    elif technique == "Ramp Up Evenly":
        mode = st.radio("Calculation Mode", ["Auto", "Manual"], horizontal=True)
        ramp_type = st.radio("Ramp Type", ["Linear", "Compound"], horizontal=True)
        tech_params['mode'] = mode.lower()
        tech_params['ramp_type'] = ramp_type.lower()
        if mode == "Manual":
            if ramp_type == "Linear":
                tech_params['increment'] = st.number_input("Interval Increment Value", value=10.0)
            else:
                tech_params['growth_factor'] = st.number_input("Ramp up Factor (e.g. 1.10 for 10%)", value=1.10, step=0.01)
                
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4 ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 04</div><div class="step-heading">Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop Enterprise Data (CSV or Excel)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- RAMP UP CONTROLLER LOGIC ---
def ramp_up_controller(demand, periods, mode="auto", ramp_type="linear", increment=None, growth_factor=None):
    baseline = demand[-1]
    if mode == "manual":
        if ramp_type == "linear":
            return [baseline + (k * (increment if increment else 0)) for k in range(1, periods + 1)]
        if ramp_type == "compound":
            return [baseline * ((growth_factor if growth_factor else 1.0) ** k) for k in range(1, periods + 1)]
    if mode == "auto":
        if len(demand) < 2: return [baseline] * periods
        if ramp_type == "linear":
            inc = (demand[-1] - demand[0]) / (len(demand) - 1)
            return [baseline + (k * inc) for k in range(1, periods + 1)]
        if ramp_type == "compound":
            if demand[0] == 0: return [baseline] * periods
            gf = (demand[-1] / demand[0]) ** (1 / (len(demand) - 1))
            return [baseline * (gf ** k) for k in range(1, periods + 1)]
    return [baseline] * periods

# --- CORE LOGIC ---
def calculate_excel_baseline(demand, tech, params, periods=1):
    if len(demand) == 0: return [0] * periods
    if tech == "Historical Average": return [np.mean(demand)] * periods
    elif tech == "Moving Average":
        n = params.get('n', 7)
        val = np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
        return [val] * periods
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.5, 0.5]))
        n = len(w)
        val = np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
        return [val] * periods
    elif tech == "Ramp Up Evenly":
        return ramp_up_controller(demand, periods, mode=params.get('mode'), ramp_type=params.get('ramp_type'), 
                                 increment=params.get('increment'), growth_factor=params.get('growth_factor'))
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return [forecast] * periods
    return [np.mean(demand)] * periods

# --- EXECUTION ---
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        model_col, part_no_col = raw.columns[0], raw.columns[1]
        date_cols = [c for c in raw.columns[4:] if pd.notnull(pd.to_datetime(c, errors='coerce', dayfirst=True))]
        df_long = raw.melt(id_vars=[model_col, part_no_col], value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "System-wide Aggregate"
        else:
            if sub_choice == "Model Wise":
                selected = st.selectbox("🎯 Target Model", df_long[model_col].unique())
                target_df = df_long[df_long[model_col] == selected].groupby('Date')['qty'].sum().reset_index()
                item_name = str(selected)
            else:
                selected_model = st.selectbox("Select Model", df_long[model_col].unique())
                selected_part = st.selectbox("Select Part No", df_long[df_long[model_col] == selected_model][part_no_col].unique())
                target_df = df_long[(df_long[model_col] == selected_model) & (df_long[part_no_col] == selected_part)].copy()
                item_name = f"{selected_part}"

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("RUN PREDICTIVE ANALYSIS"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### 🛠 Operational Viewport")
            cx1, cx2 = st.columns(2)
            with cx1: dynamic_val = st.number_input("Forecast Length", min_value=1, value=15)
            with cx2: dynamic_unit = st.selectbox("View Period", ["Days", "Weeks", "Months", "Original Selection"])
            
            # Setup Future Timeline
            last_date, last_qty = target_df['Date'].max(), target_df['qty'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=dynamic_val+1, freq=res_map[interval])[1:]
            
            # Baseline Logic
            history = target_df['qty'].tolist()
            excel_calc_col = calculate_excel_baseline(history, technique, tech_params, periods=dynamic_val)
            
            # AI Logic (Training on last available baseline value for trend adjustment)
            static_baseline = excel_calc_col[0] 
            target_df['month'], target_df['dow'] = target_df['Date'].dt.month, target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - static_baseline
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            f_df = pd.DataFrame({'Date': future_dates, 'month': future_dates.month, 'dow': future_dates.dayofweek})
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            predicted_calc_col = [round(max(b + r, 0), 2) for b, r in zip(excel_calc_col, ai_residuals)]
            excel_calc_col = [round(b, 2) for b in excel_calc_col]

            # --- DYNAMIC WINDOWING FOR GRAPH ---
            lookback = tech_params.get('n', 7) if technique == "Moving Average" else (len(tech_params.get('weights', [])) if technique == "Weightage Average" else 10)
            graph_history = target_df.tail(lookback)

            st.subheader(f"📈 Predictive Trend Analysis: {item_name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=graph_history['Date'], y=graph_history['qty'], name="History", mode='lines+markers', line=dict(color="#1a8cff", width=2.5, shape='spline'), marker=dict(size=6, color="white", line=dict(color="#1a8cff", width=1.5))))
            
            f_dates_conn = [last_date] + list(future_dates)
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty] + excel_calc_col, name="Excel Calculated Forecast", mode='lines+markers', line=dict(color="#999999", width=1.2, dash='dot', shape='spline'), marker=dict(size=4, color="#999999")))
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty] + predicted_calc_col, name="AI Predicted Forecast", mode='lines+markers', line=dict(color="#ffcc00", width=2.5, dash='dash', shape='spline'), marker=dict(size=5, color="white", line=dict(color="#ffcc00", width=1.5))))

            fig.add_vline(x=last_date, line_width=1.5, line_color="#cccccc")
            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📉 AI Pattern Adjustment")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, name="AI Adjustment", marker_color="#00B0F0"))
            fig_wig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig_wig, use_container_width=True)

            st.markdown("#### Demand Schedule")
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Predicted Forecast": predicted_calc_col, "Excel Calculated Forecast": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer: res_df.to_excel(writer, index=False)
            st.download_button("📥 EXPORT FORECAST REPORT", output.getvalue(), f"AI_Report_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Critical System Error: {e}")
