import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. ENHANCED UI & CUSTOM CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", page_icon="📈")

# Professional Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Global Card Style */
    .section-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #ececf1;
    }
    
    /* Section Headers */
    .section-title {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .step-badge {
        background-color: #007bff;
        color: white;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    /* Execute Button Styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        padding: 15px 0px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,123,255,0.2);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,123,255,0.3);
        color: white;
    }

    /* Dynamic Control Box */
    .control-panel {
        background-color: #f1f5f9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 2rem;
    }

    /* Dataframe Styling */
    .stDataFrame {
        border: 1px solid #ececf1;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #0f172a; margin-bottom: 0px;'>📊 AI-Powered Supply Chain</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;'>Precision Demand Forecasting Engine</p>", unsafe_allow_html=True)

# --- STEP 1: SCOPE ---
with st.container():
    st.markdown('<div class="section-container"><div class="section-title"><span class="step-badge">1</span> Forecasting Scope</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        main_choice = st.radio("Primary Selection Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
    with col2:
        sub_choice = None
        if main_choice == "Product Wise":
            sub_choice = st.radio("Specific Level", ["Model Wise", "Part No Wise"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: TIMELINE ---
with st.container():
    st.markdown('<div class="section-container"><div class="section-title"><span class="step-badge">2</span> Timeline Configuration</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
    with col_b:
        horizon_label = st.selectbox("Default Forecast Horizon (Initial)", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUES ---
with st.container():
    st.markdown('<div class="section-container"><div class="section-title"><span class="step-badge">3</span> Strategy & AI Technique</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        technique = st.selectbox("Excel Strategy (Baseline)", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

    tech_params = {}
    with col_d:
        if technique == "Weightage Average":
            w_in = st.text_input("Manual Weights (comma separated)", "0.2, 0.3, 0.5")
            try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
            except: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
        elif technique == "Moving Average":
            tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
        elif technique == "Ramp Up Evenly":
            tech_params['ramp_factor'] = st.number_input("Growth Factor (Multiplier)", 1.0, 2.0, 1.05)
        elif technique == "Exponentially":
            tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
    st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
with st.container():
    st.markdown('<div class="section-container"><div class="section-title"><span class="step-badge">4</span> Data Ingestion</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV/Excel (Dates as Columns)", type=['xlsx', 'csv'])
    st.markdown('</div>', unsafe_allow_html=True)

# --- CORE CALCULATION LOGIC (NO CHANGE) ---
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

# --- 7. EXECUTION ---
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
            item_name = "Aggregate Sum"
        else:
            # Better selection dropdown
            st.info(f"Select a specific {sub_choice if sub_choice else 'item'} below:")
            selected = st.selectbox(f"Select Target", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.write("---")
        if st.button("🚀 EXECUTE AI TREND ANALYSIS"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            
            # --- DYNAMIC HORIZON BOX ---
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### 🔄 Adjust Prediction Horizon")
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Look ahead amount", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Time Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculation Logic
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base_scalar
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            last_date = target_df['Date'].max()
            last_qty = target_df['qty'].iloc[-1]
            
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
            
            excel_calc_col = []
            predicted_calc_col = []
            
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- 8. TREND GRAPH ---
            st.subheader(f"📈 Trend Forecast: {item_name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Traded (History)",
                                   line=dict(color="#1a8cff", width=3, shape='spline')))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            fig.add_trace(go.Scatter(x=f_dates_conn, y=f_excel_conn, name="Excel Baseline",
                                   line=dict(color="#94a3b8", width=2, dash='dot', shape='spline')))
            fig.add_trace(go.Scatter(x=f_dates_conn, y=f_pred_conn, name="AI Predicted",
                                   line=dict(color="#f59e0b", width=4, shape='spline')))

            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig, use_container_width=True)

            # --- 9. AI WIGGLE CHART ---
            st.subheader("📉 AI Correction Patterns")
            fig_wig = go.Figure(go.Bar(x=future_dates, y=ai_residuals, marker_color="#38bdf8"))
            fig_wig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- 10. DATA TABLE & DOWNLOAD ---
            st.subheader("📋 Forecast Summary Table")
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "AI Prediction": predicted_calc_col,
                "Excel Baseline": excel_calc_col
            })
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button(label="📥 Download Export Data", data=output.getvalue(), file_name=f"AI_Forecast_{item_name}.xlsx")

    except Exception as e:
        st.error(f"Error processing file: {e}")
