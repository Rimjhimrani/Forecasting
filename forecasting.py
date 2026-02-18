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

# --- HELPER: DATE FORMATTER ---
def format_dates(series, interval):
    """Formats datetime series into user-friendly strings based on interval"""
    if interval == "Hourly": return series.dt.strftime('%d %b, %H:00')
    if interval == "Daily": return series.dt.strftime('%d-%b-%Y')
    if interval == "Weekly": return "Wk " + series.dt.isocalendar().week.astype(str) + " (" + series.dt.year.astype(str) + ")"
    if interval == "Monthly": return series.dt.strftime('%b %y')
    if interval == "Quarterly": return "Q" + series.dt.quarter.astype(str) + " " + series.dt.year.astype(str)
    if interval == "Year": return "Year " + series.dt.year.astype(str)
    return series.dt.strftime('%Y-%m-%d')

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
            '<div class="step-label">Step 02</div><div class="step-heading">Time Parameters</div>', unsafe_allow_html=True)

st.caption("📅 **Forecast Horizon (Future)**")
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=3)
with c2:
    horizon_unit = st.selectbox("Forecast Horizon Unit", ["Day(s)", "Week(s)", "Month(s)", "Quarter(s)", "Year(s)"], index=2)
with c3:
    horizon_val = st.number_input("Duration", min_value=1, value=6, key="hor_val")

st.write("")
st.caption("🕒 **Historical Lookback (Past Data)**")
h1, h2, h3 = st.columns([2, 2, 1])
with h1:
    hist_scope = st.selectbox("Historical Data Range", ["All Available Data", "Custom Lookback"])
with h2:
    if hist_scope == "Custom Lookback":
        hist_unit = st.selectbox("Lookback Unit", ["Day(s)", "Week(s)", "Month(s)", "Quarter(s)", "Year(s)"], index=4)
    else: st.info("Using 100% of data.")
with h3:
    if hist_scope == "Custom Lookback":
        hist_val = st.number_input("Duration", min_value=1, value=2, key="hist_val")

def calculate_total_periods(h_unit, h_val, interval):
    i_map = {"Hourly": 1/24, "Daily": 1, "Weekly": 7, "Monthly": 30.44, "Quarterly": 91.25, "Year": 365.25}
    h_map = {"Day(s)": 1, "Week(s)": 7, "Month(s)": 30.44, "Quarter(s)": 91.25, "Year(s)": 365.25}
    return int(np.ceil((h_map[h_unit] * h_val) / i_map[interval]))

total_forecast_periods = calculate_total_periods(horizon_unit, horizon_val, interval)
st.caption(f"💡 System will generate **{total_forecast_periods} {interval} periods**.")
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUES ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 03</div><div class="step-heading">Forecast Techniques</div>', unsafe_allow_html=True)
c4, c5 = st.columns(2)
with c4:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c5:
    tech_params = {}
    if technique == "Weightage Average":
        w_lookback = st.number_input(f"Lookback Periods", 1, 100, 3)
        tech_params["weights"] = np.ones(w_lookback) / w_lookback
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input(f"Lookback Window", min_value=1, max_value=100, value=3)
    elif technique == "Ramp Up Evenly":
        tech_params.update({'mode': 'auto', 'ramp_type': 'linear'})
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 04</div><div class="step-heading">Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop Data (CSV or Excel)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC FUNCTIONS ---
def ramp_up_controller(demand, periods, mode="auto", ramp_type="linear"):
    baseline = demand[-1] if len(demand) > 0 else 0
    if len(demand) < 2: return [baseline] * periods
    inc = (demand[-1] - demand[0]) / (len(demand) - 1)
    return [baseline + (k * inc) for k in range(1, periods + 1)]

def calculate_excel_baseline(demand, tech, params, periods=1):
    if len(demand) == 0: return [0] * periods
    if tech == "Historical Average": return [np.mean(demand)] * periods
    elif tech == "Moving Average":
        n = params.get('n', 3)
        return [np.mean(demand[-n:])] * periods
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([1]))
        n = len(w)
        val = np.dot(demand[-n:], w) / np.sum(w) if len(demand) >= n else np.mean(demand)
        return [val] * periods
    elif tech == "Ramp Up Evenly":
        return ramp_up_controller(demand, periods)
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        f = demand[0]
        for d in demand[1:]: f = alpha * d + (1 - alpha) * f
        return [f] * periods
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
        df_long = df_long.sort_values('Date')

        # Scope Filtering
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Total Aggregate"
        else:
            if sub_choice == "Model Wise":
                sel = st.selectbox("Select Model", df_long[model_col].unique())
                target_df = df_long[df_long[model_col] == sel].groupby('Date')['qty'].sum().reset_index()
                item_name = str(sel)
            else:
                m_sel = st.selectbox("Model", df_long[model_col].unique())
                p_sel = st.selectbox("Part", df_long[df_long[model_col] == m_sel][part_no_col].unique())
                target_df = df_long[(df_long[model_col] == m_sel) & (df_long[part_no_col] == p_sel)].copy()
                item_name = str(p_sel)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Year": "YS"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        # Lookback Filter
        if hist_scope == "Custom Lookback":
            h_map = {"Day(s)": 1, "Week(s)": 7, "Month(s)": 30, "Quarter(s)": 91, "Year(s)": 365}
            cutoff = target_df['Date'].max() - timedelta(days=h_map[hist_unit]*hist_val)
            target_df = target_df[target_df['Date'] >= cutoff].reset_index(drop=True)

        if st.button("RUN PREDICTIVE ANALYSIS"):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            
            # Future Dates Generation
            last_date, last_qty = target_df['Date'].max(), target_df['qty'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=total_forecast_periods+1, freq=res_map[interval])[1:]
            
            # Calculations
            history = target_df['qty'].tolist()
            baseline_vals = calculate_excel_baseline(history, technique, tech_params, periods=total_forecast_periods)
            
            # Simple AI Shift
            target_df['month'], target_df['dow'] = target_df['Date'].dt.month, target_df['Date'].dt.dayofweek
            model = XGBRegressor().fit(target_df[['month', 'dow']], target_df['qty'] - np.mean(history))
            f_feat = pd.DataFrame({'month': future_dates.month, 'dow': future_dates.dayofweek})
            ai_vals = [round(max(b + r, 0), 2) for b, r in zip(baseline_vals, model.predict(f_feat))]

            # FORMAT DATES FOR DISPLAY
            target_df['DisplayDate'] = format_dates(target_df['Date'], interval)
            future_display_dates = format_dates(pd.Series(future_dates), interval).tolist()

            # Charting
            fig = go.Figure()
            # History
            fig.add_trace(go.Scatter(x=target_df['DisplayDate'], y=target_df['qty'], name="History", line=dict(color="#1a8cff")))
            # Forecasts
            all_f_dates = [target_df['DisplayDate'].iloc[-1]] + future_display_dates
            fig.add_trace(go.Scatter(x=all_f_dates, y=[last_qty] + [round(b,2) for b in baseline_vals], name="Baseline", line=dict(dash='dot', color="#94a3b8")))
            fig.add_trace(go.Scatter(x=all_f_dates, y=[last_qty] + ai_vals, name="AI Adjusted", line=dict(width=4, color="#4F46E5")))

            fig.update_layout(template="plotly_white", hovermode="x unified", height=400, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Schedule Table
            st.markdown("#### Forecast Schedule")
            res_df = pd.DataFrame({
                "Period": future_display_dates,
                "AI Predicted Qty": ai_vals,
                "Baseline Qty": [round(b, 2) for b in baseline_vals]
            })
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer: res_df.to_excel(writer, index=False)
            st.download_button("📥 DOWNLOAD REPORT", output.getvalue(), f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Execution Error: {e}")
