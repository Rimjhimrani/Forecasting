import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & CLEAN WHITE CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", page_icon="📊")

st.markdown("""
<style>
    /* Global Background and Font */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Global text color for readability */
    html, body, [class*="css"] {
        color: #1e293b;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Container for each step (The "Card") */
    .main-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* Titles and Labels */
    .step-label {
        color: #2563eb;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        display: block;
    }
    
    .step-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 20px;
    }

    /* Big Green Execute Button */
    .stButton > button {
        width: 100% !important;
        background-color: #10b981 !important;
        color: white !important;
        border: none !important;
        padding: 15px !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #059669 !important;
    }

    /* Secondary Forecast Control Box */
    .control-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("📊 Supply Chain Precision Forecast")
st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Combine statistical baselines with AI-driven pattern recognition.</p>", unsafe_allow_html=True)
st.divider()

# --- STEP 1 & 2: SCOPE & TIMELINE ---
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-label">Step 1</span><div class="step-title">Forecasting Scope</div>', unsafe_allow_html=True)
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_s2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-label">Step 2</span><div class="step-title">Timeline Settings</div>', unsafe_allow_html=True)
    interval = st.selectbox("Interval Frequency", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    horizon_label = st.selectbox("Initial Horizon", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3 & 4: STRATEGY & UPLOAD ---
col_s3, col_s4 = st.columns(2)

with col_s3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-label">Step 3</span><div class="step-title">Strategy Selection</div>', unsafe_allow_html=True)
    technique = st.selectbox("Statistical Baseline", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
    
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weights", "0.3, 0.7")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.5])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback Window", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
    st.markdown('</div>', unsafe_allow_html=True)

with col_s4:
    st.markdown('<div class="main-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<span class="step-label">Step 4</span><div class="step-title">Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop CSV or Excel File", type=['xlsx', 'csv'])
    st.markdown('</div>', unsafe_allow_html=True)

# --- CORE CALCULATION LOGIC ---
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
            selected = st.selectbox("🎯 Select Product/Model to Forecast", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.write("")
        if st.button("🚀 GENERATE AI FORECAST"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
            
            # --- DYNAMIC HORIZON CONTROL ---
            st.markdown('<div class="control-box">', unsafe_allow_html=True)
            st.markdown("<strong>🛠 Adjust Results Horizon</strong>", unsafe_allow_html=True)
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Lookahead Qty", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Time Metric", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculation Core
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
            st.subheader(f"📈 Forecast Analysis for: {item_name}")
            fig = go.Figure()
            # History
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Traded History", line=dict(color="#0f172a", width=2.5, shape='spline')))
            
            f_dates_conn = [last_date] + list(future_dates)
            # Baseline
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty] + excel_calc_col, name="Excel Baseline", line=dict(color="#94a3b8", width=1.5, dash='dot')))
            # AI Forecast
            fig.add_trace(go.Scatter(x=f_dates_conn, y=[last_qty] + predicted_calc_col, name="AI Prediction", line=dict(color="#2563eb", width=3.5, shape='spline')))
            
            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # AI Wiggles (Bar Chart)
            st.subheader("📉 AI Pattern Adjustments (Residuals)")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, marker_color="#3b82f6", name="AI Correction"))
            fig_wig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- TABLE & DOWNLOAD ---
            st.subheader("📋 Forecast Summary")
            res_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "AI Prediction": predicted_calc_col,
                "Excel Baseline": excel_calc_col
            })
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button(label="📥 Export Result to Excel", data=output.getvalue(), file_name=f"Forecast_{item_name}.xlsx")

    except Exception as e:
        st.error(f"Error occurred during processing: {e}")
