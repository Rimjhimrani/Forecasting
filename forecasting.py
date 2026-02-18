import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io
from datetime import timedelta

if "wa_weights" not in st.session_state:
    st.session_state.wa_weights = None

# --- 1. PREMIUM ENTERPRISE UI CONFIG ---
st.set_page_config(page_title="AgiloForecast | Enterprise AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #FFFFFF; color: #111827; }
    .block-container { padding-top: 5rem; max-width: 1000px; margin: 0 auto; }
    .step-wrapper { position: relative; padding-left: 40px; margin-bottom: 50px; border-left: 2px solid #E5E7EB; }
    .step-dot { position: absolute; left: -9px; top: 0; width: 16px; height: 16px; background-color: #FFFFFF; border: 2px solid #4F46E5; border-radius: 50%; }
    .step-label { color: #4F46E5; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .step-heading { font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 20px; }
    div.stButton > button { width: 100%; background-color: #111827 !important; color: #FFFFFF !important; border: none !important; padding: 18px !important; font-size: 1.1rem !important; font-weight: 600 !important; border-radius: 10px !important; }
    .insight-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 30px; border-radius: 20px; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div style="text-align: center; margin-bottom: 80px;">'
            '<h1 style="font-size: 3rem; font-weight: 800; letter-spacing: -2px; color: #4F46E5;">Agilo<span style="color:#111827;">Forecast</span></h1>'
            '<p style="color: #6B7280; font-size: 1.2rem;">AI-Driven Intelligence for Forecasting & Decisions</p>'
            '</div>', unsafe_allow_html=True)

# --- HELPER: LABEL LOGIC ---
def get_display_labels(dates, interval, is_forecast=False):
    dates_ser = pd.Series(dates)
    if is_forecast:
        if interval == "Daily": 
            return dates_ser.dt.strftime('%d-%m-%Y').tolist()
        if interval == "Weekly": 
            return [f"Week {i+1}" for i in range(len(dates_ser))]
        if interval == "Hourly": 
            return [f"Hr {i+1}" for i in range(len(dates_ser))]
    
    if interval == "Monthly": return dates_ser.dt.strftime('%b %y').tolist()
    if interval == "Quarterly": return [f"Q{d.quarter}-{str(d.year)[2:]}" for d in dates_ser]
    if interval == "Year": return [f"Year {str(d.year)[2:]}" for d in dates_ser]
    return dates_ser.dt.strftime('%d-%m-%Y').tolist()

# --- STEP 1: SCOPE ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 01</div><div class="step-heading">Forecasting Scope</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Resolution Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: PARAMETERS ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 02</div><div class="step-heading">Set Parameters</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c2:
    horizon_unit = st.selectbox("Forecast Horizon Unit", ["Day(s)", "Week(s)", "Month(s)", "Quarter(s)", "Year(s)"], index=2)
with c3:
    horizon_val = st.number_input("Horizon Duration", min_value=1, value=1)

st.write("")
st.caption("🕒 **Historical Lookback (Past Data Context)**")
h1, h2, h3 = st.columns([2, 2, 1])
with h1:
    hist_scope = st.selectbox("History Range", ["All Available Data", "Custom Lookback"])
with h2:
    if hist_scope == "Custom Lookback":
        hist_unit = st.selectbox("Lookback Unit", ["Day(s)", "Week(s)", "Month(s)", "Quarter(s)", "Year(s)"], index=2)
with h3:
    if hist_scope == "Custom Lookback":
        hist_val = st.number_input("Lookback Duration", min_value=1, value=6)

def calculate_total_periods(inv, h_unit, h_val):
    interval_to_days = {"Hourly": 1/24, "Daily": 1, "Weekly": 7, "Monthly": 30, "Quarterly": 91, "Year": 365}
    horizon_to_days = {"Day(s)": 1, "Week(s)": 7, "Month(s)": 30, "Quarter(s)": 91, "Year(s)": 365}
    return int(np.ceil((horizon_to_days[h_unit] * h_val) / interval_to_days[inv]))

total_forecast_periods = calculate_total_periods(interval, horizon_unit, horizon_val)
st.caption(f"💡 System will generate **{total_forecast_periods}** {interval} data points.")
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUES ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 03</div><div class="step-heading">Forecast Techniques</div>', unsafe_allow_html=True)
c4, c5 = st.columns(2)
unit_map = {"Hourly": "Hours", "Daily": "Days", "Weekly": "Weeks", "Monthly": "Months", "Quarterly": "Quarters", "Year": "Years"}
current_unit = unit_map.get(interval, "Periods")

with c4:
    technique = st.selectbox("Forecast Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c5:
    tech_params = {}
    if technique == "Weightage Average":
        w_mode = st.radio("Weight Configuration", ["Manual Entry", "Automated (Evenly)"], horizontal=True)
        if w_mode == "Manual Entry":
            w_in = st.text_input("Manual Ratios (comma separated)", value="0.3,0.7")
            try: weights = np.array([float(x.strip()) for x in w_in.split(",")])
            except: weights = np.array([0.5, 0.5])
        else:
            w_lookback = st.number_input(f"Lookback ({current_unit})", 1, 100, 3)
            weights = np.ones(w_lookback) / w_lookback
        tech_params["weights"] = weights
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input(f"Lookback Window ({current_unit})", min_value=1, max_value=100, value=7)
    elif technique == "Ramp Up Evenly":
        mode = st.radio("Calculation Mode", ["Auto", "Manual"], horizontal=True)
        ramp_type = st.radio("Ramp Type", ["Linear", "Compound"], horizontal=True)
        tech_params.update({'mode': mode.lower(), 'ramp_type': ramp_type.lower()})
        if mode == "Manual":
            if ramp_type == "Linear": tech_params['increment'] = st.number_input("Interval Increment Value", value=10.0)
            else: tech_params['growth_factor'] = st.number_input("Ramp up Factor (e.g. 1.10)", value=1.10)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 04</div><div class="step-heading">Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop Enterprise Data (CSV or Excel)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC FUNCTIONS (ROUNDED UP) ---
def ramp_up_controller(demand, periods, mode="auto", ramp_type="linear", increment=None, growth_factor=None):
    baseline = demand[-1]
    if mode == "manual":
        if ramp_type == "linear": return [baseline + (k * (increment or 0)) for k in range(1, periods + 1)]
        if ramp_type == "compound": return [baseline * ((growth_factor or 1.0) ** k) for k in range(1, periods + 1)]
    if mode == "auto":
        if len(demand) < 2: return [baseline] * periods
        if ramp_type == "linear":
            inc = (demand[-1] - demand[0]) / (len(demand) - 1)
            return [baseline + (k * inc) for k in range(1, periods + 1)]
        if ramp_type == "compound":
            gf = (demand[-1] / demand[0]) ** (1 / (len(demand) - 1)) if demand[0] != 0 else 1.0
            return [baseline * (gf ** k) for k in range(1, periods + 1)]
    return [baseline] * periods

def calculate_excel_baseline(demand, tech, params, periods=1):
    if len(demand) == 0: return [0] * periods
    if tech == "Historical Average": res = [np.mean(demand)] * periods
    elif tech == "Moving Average":
        n = params.get('n', 7)
        val = np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
        res = [val] * periods
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.5, 0.5]))
        n = len(w)
        val = np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
        res = [val] * periods
    elif tech == "Ramp Up Evenly":
        res = ramp_up_controller(demand, periods, **{k: v for k, v in params.items() if k in ['mode', 'ramp_type', 'increment', 'growth_factor']})
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        res = [forecast] * periods
    else:
        res = [np.mean(demand)] * periods
    
    # ROUND UP BASELINE
    return [int(np.ceil(x)) for x in res]

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
            item_name = "Aggregate"
        else:
            if sub_choice == "Model Wise":
                selected = st.selectbox("🎯 Target Model", df_long[model_col].unique())
                target_df = df_long[df_long[model_col] == selected].groupby('Date')['qty'].sum().reset_index()
                item_name = str(selected)
            else:
                sm = st.selectbox("Select Model", df_long[model_col].unique())
                sp = st.selectbox("Select Part No", df_long[df_long[model_col] == sm][part_no_col].unique())
                target_df = df_long[(df_long[model_col] == sm) & (df_long[part_no_col] == sp)].copy()
                item_name = str(sp)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Year": "YS"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if hist_scope == "Custom Lookback":
            h_lookup = {"Day(s)": 1, "Week(s)": 7, "Month(s)": 30, "Quarter(s)": 91, "Year(s)": 365}
            cutoff = target_df['Date'].max() - timedelta(days=h_lookup[hist_unit]*hist_val)
            target_df = target_df[target_df['Date'] >= cutoff].reset_index(drop=True)

        if st.button("RUN PREDICTIVE ANALYSIS"):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            
            periods_to_forecast = total_forecast_periods
            last_date, last_qty = target_df['Date'].max(), target_df['qty'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=periods_to_forecast+1, freq=res_map[interval])[1:]
            
            history = target_df['qty'].tolist()
            excel_calc_col = calculate_excel_baseline(history, technique, tech_params, periods=periods_to_forecast)
            
            # AI Pattern Logic (Residual Formula)
            static_baseline = excel_calc_col[0] 
            target_df['month'], target_df['dow'] = target_df['Date'].dt.month, target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - static_baseline
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            f_df = pd.DataFrame({'Date': future_dates, 'month': future_dates.month, 'dow': future_dates.dayofweek})
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            # ROUND UP FINAL AI PREDICTION
            predicted_calc_col = [int(np.ceil(max(b + r, 0))) for b, r in zip(excel_calc_col, ai_residuals)]

            hist_labels = get_display_labels(target_df['Date'], interval, is_forecast=False)
            future_labels = get_display_labels(future_dates, interval, is_forecast=True)

            # Graphing
            st.subheader(f"📈 {interval} Trend Analysis: {item_name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_labels, y=target_df['qty'], name="History", line=dict(color="#1a8cff", width=2)))
            conn_x = [hist_labels[-1]] + list(future_labels)
            fig.add_trace(go.Scatter(x=conn_x, y=[last_qty] + excel_calc_col, name="Excel Forecast", line=dict(color="#999999", dash='dot')))
            fig.add_trace(go.Scatter(x=conn_x, y=[last_qty] + predicted_calc_col, name="AI Predicted Forecast", line=dict(color="#4F46E5", width=3)))

            fig.update_layout(template="plotly_white", hovermode="x unified", height=450, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Demand Schedule")
            res_df = pd.DataFrame({"Period": future_labels, "AI Predicted Forecast": predicted_calc_col, "Excel Calculated Forecast": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer: res_df.to_excel(writer, index=False)
            st.download_button("📥 EXPORT REPORT", output.getvalue(), f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
