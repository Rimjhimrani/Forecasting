import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. UI SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f7f6; }
    .step-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border-left: 6px solid #1a8cff;
    }
    .step-header {
        color: #1e3d59;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .step-number {
        background-color: #1a8cff;
        color: white;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 15px;
        font-size: 18px;
        font-weight: bold;
    }
    .execute-btn > button {
        width: 100% !important;
        background-color: #00B050 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 18px !important;
        border-radius: 10px !important;
        border: none !important;
        font-size: 20px !important;
    }
    .dynamic-box {
        background-color: #fff9db;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #fab005;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI-Powered Supply Chain Precision Forecast")

# --- 2. STEP 1: SCOPE ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">1</div>Forecasting Scope Selection</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Primary Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with col2:
    sub_choice = None
    if main_choice == "Product Wise":
        sub_choice = st.radio("Level", ["Model Wise", "Part No Wise"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 3. STEP 2: TIMELINE ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">2</div>Timeline Configuration</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    interval = st.selectbox("Forecast Interval", options=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Year"], index=1)
with col_b:
    horizon_label = st.selectbox("Default Horizon (Initial)", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. STEP 3: TECHNIQUES ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">3</div>Select Strategy & AI Technique</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)
with col_c:
    technique = st.selectbox("Excel Strategy (Baseline)", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])

tech_params = {}
with col_d:
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weights (comma separated)", "0.2, 0.3, 0.5")
        tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback window (n)", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Factor (Multiplier)", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- 5. STEP 4: UPLOAD ---
st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">4</div>Data Ingestion</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV/Excel (Dates as Columns)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. CORE LOGIC FUNCTIONS ---
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
            selected = st.selectbox(f"Select Target", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
            item_name = str(selected)

        res_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Year": "A"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        if st.button("🚀 EXECUTE AI TREND ANALYSIS"):
            st.session_state.run_analysis = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('run_analysis', False):
            st.divider()
            
            # --- DYNAMIC HORIZON BOX ---
            st.markdown('<div class="dynamic-box">', unsafe_allow_html=True)
            st.write("🔄 **Adjust Horizon Instantly for the Trend Chart:**")
            col_hz1, col_hz2 = st.columns(2)
            with col_hz1:
                dynamic_val = st.number_input("Enter Quantity", min_value=1, value=15)
            with col_hz2:
                dynamic_unit = st.selectbox("Select Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculation
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            # AI Model
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
            
            # Predictions
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_residuals = model.predict(f_df[['month', 'dow']])
            
            excel_calc_col = []
            predicted_calc_col = []
            
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- TREND GRAPH (FIXED ANNOTATIONS) ---
            st.subheader(f"📈 Predictive Trend Analysis: {item_name}")
            fig = go.Figure()

            # Traded Line
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Traded",
                mode='lines+markers', line=dict(color="#1a8cff", width=4, shape='spline'),
                marker=dict(size=8, color="white", line=dict(color="#1a8cff", width=2))
            ))

            # Prediction Line
            f_dates_conn = [last_date] + list(future_dates)
            f_preds_conn = [last_qty] + list(predicted_calc_col)
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_preds_conn, name="Prediction",
                mode='lines', line=dict(color="#ffcc00", width=4, dash='dash', shape='spline')
            ))

            # Vertical Line
            fig.add_vline(x=last_date, line_width=2, line_dash="solid", line_color="#cccccc")

            # Icons (Fixed properties)
            fig.add_annotation(
                x=target_df['Date'].iloc[int(len(target_df)*0.8)], 
                y=target_df['qty'].max()*1.1, 
                text="🛍️", showarrow=False, 
                bgcolor="rgba(26,140,255,0.1)", 
                bordercolor="#1a8cff", 
                borderwidth=2,
                borderpad=8
            )
            fig.add_annotation(
                x=future_dates[int(len(future_dates)*0.5)] if len(future_dates)>0 else last_date, 
                y=max(predicted_calc_col)*1.1 if len(predicted_calc_col)>0 else last_qty, 
                text="📢", showarrow=False, 
                bgcolor="rgba(255,204,0,0.1)", 
                bordercolor="#ffcc00", 
                borderwidth=2,
                borderpad=8
            )

            fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- TABLES & DOWNLOAD ---
            st.subheader("📋 Forecast Data Results")
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "Predicted Calculated Forecast": predicted_calc_col,
                "Excel Calculated Forecast": excel_calc_col
            })
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button(label="📥 Download Excel Result", data=output.getvalue(), file_name=f"Forecast_{item_name}.xlsx")

    except Exception as e:
        st.error(f"Error: {e}")
