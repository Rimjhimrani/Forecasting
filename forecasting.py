import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Clean white background */
    .main {
        background-color: #ffffff;
    }
    
    .block-container {
        padding-top: 3rem;
        max-width: 1200px;
    }
    
    /* Simple card design */
    .step-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid #4F46E5;
    }
    
    /* Header */
    .step-header {
        color: #111827;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    /* Step number */
    .step-number {
        background: #4F46E5;
        color: white;
        border-radius: 8px;
        width: 32px;
        height: 32px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 12px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Main title */
    h1 {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-bottom: 2rem !important;
        padding-bottom: 1rem !important;
        border-bottom: 3px solid #4F46E5 !important;
    }
    
    /* Section headers */
    h2, h3 {
        color: #111827 !important;
        font-weight: 600 !important;
        margin-top: 0 !important;
    }
    
    /* Execute button */
    .execute-btn > button {
        width: 100% !important;
        background: #4F46E5 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.9rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
        margin-top: 1rem !important;
    }
    
    .execute-btn > button:hover {
        background: #4338CA !important;
    }
    
    /* Dynamic controls box */
    .dynamic-box {
        background: #F3F4F6;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    
    .dynamic-box p {
        color: #374151;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #059669 !important;
        color: white !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        border: none !important;
    }
    
    .stDownloadButton > button:hover {
        background: #047857 !important;
    }
    
    /* Info box */
    .stInfo {
        background: #EEF2FF;
        border-left: 3px solid #4F46E5;
        border-radius: 6px;
    }
    
    /* Input styling */
    .stSelectbox label, .stRadio label, .stNumberInput label {
        font-weight: 500 !important;
        color: #374151 !important;
        font-size: 0.9rem !important;
    }
    
    /* Clean divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Table */
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stFileUploader:hover {
        border-color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Supply Chain Forecast")

# --- 2. STEP 1: SCOPE ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">1</div>Forecast Scope</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Selection Type", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Detail Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 3. STEP 2: TIMELINE ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">2</div>Time Settings</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    interval = st.selectbox("Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_b:
    horizon_label = st.selectbox("Forecast Period", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. STEP 3: TECHNIQUES ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">3</div>Forecast Method</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)
with col_c:
    technique = st.selectbox("Baseline Strategy", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
with col_d:
    if technique == "Weightage Average":
        w_in = st.text_input("Weights (comma separated)", "0.2, 0.3, 0.5")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.33, 0.33, 0.34])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Window Size", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Factor", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- 5. STEP 4: UPLOAD ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">4</div>Upload Data</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE CALCULATION LOGIC ---
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
            selected = st.selectbox(f"Select Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast"):
            st.session_state.run_analysis = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('run_analysis', False):
            st.divider()
            
            # --- DYNAMIC HORIZON BOX ---
            st.markdown('<div class="dynamic-box">', unsafe_allow_html=True)
            st.markdown("**Adjust Forecast Horizon**")
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Quantity", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Time Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # 1. Excel Baseline scalar
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            # 2. AI Model Training (Residuals)
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base_scalar
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            # 3. Future Dates Calculation
            last_date = target_df['Date'].max()
            last_qty = target_df['qty'].iloc[-1]
            
            if dynamic_unit == "Original Selection":
                h_map = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
                end_date = last_date + pd.Timedelta(days=h_map[horizon_label])
            elif dynamic_unit == "Days": end_date = last_date + pd.Timedelta(days=dynamic_val)
            elif dynamic_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=dynamic_val)
            else: end_date = last_date + pd.DateOffset(months=dynamic_val)
            
            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            
            # 4. Predictions Construction
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
            st.subheader(f"Forecast Trend: {item_name}")
            fig = go.Figure()

            # Historical Data
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Historical",
                mode='lines+markers', 
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=6, color="#2563EB")
            ))

            f_dates_conn = [last_date] + list(future_dates)
            f_excel_conn = [last_qty] + list(excel_calc_col)
            f_pred_conn = [last_qty] + list(predicted_calc_col)

            # Excel Baseline
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Baseline",
                mode='lines+markers', 
                line=dict(color="#9CA3AF", width=2, dash='dot'),
                marker=dict(size=5, color="#9CA3AF")
            ))

            # AI Forecast
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Forecast",
                mode='lines+markers', 
                line=dict(color="#10B981", width=2.5, dash='dash'),
                marker=dict(size=6, color="#10B981")
            ))

            fig.add_vline(x=last_date, line_width=1, line_color="#D1D5DB", line_dash="dash")

            fig.update_layout(
                template="plotly_white", 
                hovermode="x unified", 
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=12, color="#374151")
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 9. AI ADJUSTMENT CHART ---
            st.subheader("AI Pattern Adjustments")
            st.info("Shows how AI modifies the baseline based on detected patterns")
            
            colors = ['#10B981' if x >= 0 else '#EF4444' for x in ai_residuals]
            fig_wig = go.Figure(go.Bar(
                x=future_dates, 
                y=ai_residuals, 
                name="Adjustment",
                marker=dict(color=colors)
            ))
            fig_wig.update_layout(
                template="plotly_white", 
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(size=12, color="#374151"),
                yaxis=dict(title="Adjustment Value")
            )
            st.plotly_chart(fig_wig, use_container_width=True)

            # --- 10. DATA TABLE & DOWNLOAD ---
            st.subheader("Forecast Results")
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "AI Forecast": predicted_calc_col,
                "Baseline": excel_calc_col
            })
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='Forecast')
            
            st.download_button(
                label="📥 Download Excel", 
                data=output.getvalue(), 
                file_name=f"Forecast_{item_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error: {e}")
