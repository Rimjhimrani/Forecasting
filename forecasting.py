import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & MINIMALIST CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Global White Aesthetic */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Clean Title Styling */
    .title-text {
        text-align: center;
        color: #111827;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        text-align: center;
        color: #6B7280;
        margin-bottom: 3rem;
    }

    /* Vertical Step Container */
    .step-section {
        border-bottom: 1px solid #F3F4F6;
        padding: 2.5rem 0;
    }
    
    .step-badge {
        background: #F3F4F6;
        color: #1f2937;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: inline-block;
        border: 1px solid #e5e7eb;
    }
    
    .step-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1.5rem;
    }

    /* Primary Execute Button */
    div.stButton > button:first-child {
        width: 100%;
        background-color: #000000;
        color: #ffffff;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        margin-top: 2rem;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #374151;
        color: #ffffff;
    }

    /* Result Box Adjustment */
    .result-container {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #000000;
    }

    /* Dynamic Control Area */
    .control-area {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-text">AI Precision Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Follow the 4 steps below to generate your prediction.</div>', unsafe_allow_html=True)

# --- STEP 1: SCOPE ---
st.markdown('<div class="step-section"><span class="step-badge">Step 1</span><div class="step-title">Forecasting Scope</div>', unsafe_allow_html=True)
main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True, label_visibility="collapsed")
sub_choice = None
if main_choice == "Product Wise":
    st.write("Select Detail Level:")
    sub_choice = st.radio("Detail Level", ["Model Wise", "Part No Wise"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: TIMELINE ---
st.markdown('<div class="step-section"><span class="step-badge">Step 2</span><div class="step-title">Time Configuration</div>', unsafe_allow_html=True)
c_t1, c_t2 = st.columns(2)
with c_t1:
    interval = st.selectbox("Interval Frequency", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with c_t2:
    horizon_label = st.selectbox("Default Forecast Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUES ---
st.markdown('<div class="step-section"><span class="step-badge">Step 3</span><div class="step-title">Baseline Strategy</div>', unsafe_allow_html=True)
technique = st.selectbox("Statistical Method", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
if technique == "Weightage Average":
    w_in = st.text_input("Weights (comma separated)", "0.3, 0.7")
    try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
    except: tech_params['weights'] = np.array([0.5, 0.5])
elif technique == "Moving Average":
    tech_params['n'] = st.number_input("Window Size (n)", 2, 30, 7)
elif technique == "Ramp Up Evenly":
    tech_params['ramp_factor'] = st.number_input("Growth Factor (Multiplier)", 1.0, 2.0, 1.05)
elif technique == "Exponentially":
    tech_params['alpha'] = st.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-section"><span class="step-badge">Step 4</span><div class="step-title">Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel (Horizontal Format)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE LOGIC ---
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

# --- EXECUTION ---
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
            selected = st.selectbox("Select Target Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("🚀 EXECUTE AI TREND ANALYSIS"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            
            # Dynamic Control
            st.markdown('<div class="control-area"><b>Live Horizon Adjust</b>', unsafe_allow_html=True)
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                dynamic_val = st.number_input("Lookahead Qty", min_value=1, value=15)
            with col_h2:
                dynamic_unit = st.selectbox("Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Modeling
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
            st.subheader(f"Trend Chart: {item_name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Traded", line=dict(color="#111827", width=2)))
            
            f_dates_conn = [last_date] + list(future_dates)
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+excel_calc_col, name="Baseline", line=dict(color="#9CA3AF", dash='dot')))
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty]+predicted_calc_col, name="AI Prediction", line=dict(color="#2563EB", width=3)))
            
            fig.update_layout(template="plotly_white", hovermode="x unified", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # AI Residuals (Wiggles)
            st.subheader("AI Seasonal Adjustment")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, marker_color="#3B82F6"))
            fig_wig.update_layout(template="plotly_white", height=250)
            st.plotly_chart(fig_wig, use_container_width=True)

            # Data Table
            st.subheader("Results Table")
            res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "AI Forecast": predicted_calc_col, "Baseline": excel_calc_col})
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel Report", output.getvalue(), f"Forecast_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
