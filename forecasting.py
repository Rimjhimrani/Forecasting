import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- UI SETTINGS ---
st.set_page_config(page_title="AI Precision Forecast", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f4f7f6; }
    .step-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border-left: 5px solid #00B0F0;
    }
    .execute-btn > button {
        width: 100% !important;
        background-color: #00B050 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 15px !important;
        border-radius: 8px !important;
        font-size: 18px !important;
    }
    .logic-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00B0F0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI-Powered Supply Chain Precision Forecast")

# --- STEPS 1-4 (SAME AS PREVIOUS) ---
st.markdown('<div class="step-card"><b>Step 1 & 2: Scope & Interval</b>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: main_choice = st.radio("Path", ["Aggregate Wise", "Product Wise"], horizontal=True)
with c2: interval = st.selectbox("Interval", options=["Daily", "Weekly", "Monthly", "Quarterly"], index=0)
with c3: horizon_label = st.selectbox("Default Horizon", ["Week", "Month", "Quarter", "Year"], index=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="step-card"><b>Step 3 & 4: Strategy & Data</b>', unsafe_allow_html=True)
c4, c5 = st.columns(2)
with c4: technique = st.selectbox("Excel Strategy", ["Historical Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c5: uploaded_file = st.file_uploader("Upload CSV/Excel", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CALCULATION LOGIC ---
def calculate_excel_baseline(demand, tech):
    if len(demand) == 0: return 0
    if tech == "Moving Average": return np.mean(demand[-7:])
    if tech == "Exponentially": return demand[-1] * 0.3 + np.mean(demand) * 0.7
    return np.mean(demand)

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

        if main_choice == "Product Wise":
            selected = st.selectbox(f"Select Item", df_long[id_col].unique())
            target_df = df_long[df_long[id_col] == selected].copy()
        else:
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()

        res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        st.markdown('<div class="execute-btn">', unsafe_allow_html=True)
        if st.button("🚀 EXECUTE HYBRID AI FORECAST"):
            st.session_state.run = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('run', False):
            # --- DYNAMIC HORIZON BOX ---
            st.markdown('<div style="background-color:#fff9db; padding:15px; border-radius:10px; border:1px solid #fab005;">', unsafe_allow_html=True)
            col_z1, col_z2 = st.columns(2)
            with col_z1: d_val = st.number_input("Forecast Length", min_value=1, value=30)
            with col_z2: d_unit = st.selectbox("Horizon Unit", ["Days", "Weeks", "Months", "Original Selection"])
            st.markdown('</div>', unsafe_allow_html=True)

            # 1. Excel Calculation
            history = target_df['qty'].tolist()
            excel_base = calculate_excel_baseline(history, technique)
            
            # 2. AI Adjustment Calculation (Learning patterns)
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['diff'] = target_df['qty'] - excel_base
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            model.fit(target_df[['month', 'dow']], target_df['diff'])
            
            # 3. Future Timeline
            last_date = target_df['Date'].max()
            if d_unit == "Days": end_date = last_date + pd.Timedelta(days=d_val)
            elif d_unit == "Weeks": end_date = last_date + pd.Timedelta(weeks=d_val)
            elif d_unit == "Months": end_date = last_date + pd.DateOffset(months=d_val)
            else: 
                h_map = {"Week": 7, "Month": 30, "Quarter": 90, "Year": 365}
                end_date = last_date + pd.Timedelta(days=h_map[horizon_label])

            future_dates = pd.date_range(start=last_date, end=end_date, freq=res_map[interval])[1:]
            
            # 4. Final Prediction Construction
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_pattern_adjustment = model.predict(f_df[['month', 'dow']])
            
            excel_forecast_column = [round(excel_base, 2)] * len(future_dates)
            final_ai_forecast_column = [round(max(excel_base + adj, 0), 2) for adj in ai_pattern_adjustment]

            # --- VISUALIZING THE AI LOGIC ---
            st.subheader("1. Final Forecast (Baseline + AI Adjustment)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="History", line=dict(color="#1e3d59")))
            fig1.add_trace(go.Scatter(x=future_dates, y=excel_forecast_column, name="Excel Baseline (Flat)", line=dict(color="gray", dash='dash')))
            fig1.add_trace(go.Scatter(x=future_dates, y=final_ai_forecast_column, name="AI Enhanced Forecast", line=dict(color="#FF8C00", width=3)))
            fig1.update_layout(template="plotly_white", hovermode="x unified", height=400)
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("2. AI Pattern Breakdown (The 'Wiggles')")
            st.info("This chart shows exactly how much the AI is adding or subtracting from the Excel average based on seasonality.")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=future_dates, y=ai_pattern_adjustment, name="AI Adjustment Amount", marker_color="#00B0F0"))
            fig2.update_layout(template="plotly_white", height=250, title="Positive/Negative adjustments identified by AI")
            st.plotly_chart(fig2, use_container_width=True)

            # --- DOWNLOAD TABLE ---
            download_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "Excel Calculated Forecast": excel_forecast_column,
                "AI Pattern Adjustment": np.round(ai_pattern_adjustment, 2),
                "Predicted Calculated Forecast": final_ai_forecast_column
            })
            
            st.subheader("📋 Forecast Breakdown Table")
            st.dataframe(download_df, use_container_width=True, hide_index=True)

            # Excel Export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name='Forecast_Details')
            st.download_button("📥 Download Excel Report", data=output.getvalue(), file_name="AI_Forecast_Report.xlsx")

    except Exception as e:
        st.error(f"Error: {e}")
