import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. PAGE CONFIG & MODERN STYLING ---
st.set_page_config(page_title="AI Order Forecast", layout="wide")

st.markdown("""
<style>
    /* Global Styles */
    .main { background-color: #fcfcfd; }
    h1, h2, h3 { color: #1e293b; font-family: 'Inter', sans-serif; }
    
    /* Card Container */
    .app-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Header Gradient */
    .header-gradient {
        background: linear-gradient(90deg, #4f46e5 0%, #0ea5e9 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 10px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# --- 2. HEADER ---
st.markdown("""
<div class="header-gradient">
    <h1 style="color: white; margin-bottom: 5px;">📦 AI Order Forecasting</h1>
    <p style="opacity: 0.9; font-size: 1.1rem;">Turn historical data into intelligent supply chain insights</p>
</div>
""", unsafe_allow_html=True)

# --- 3. STEP 1: DATA UPLOAD ---
st.markdown("### 1. Upload Historical Data")
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Excel or CSV (Wide Format: Date Columns across the top)", type=['xlsx', 'csv'], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. STEP 2: CONFIGURATION (Only shows if file is uploaded) ---
if uploaded_file:
    try:
        # Load Data
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        id_col = df_raw.columns[0]
        date_cols = [c for c in df_raw.columns[1:] if pd.to_datetime(c, errors='coerce') is not pd.NaT]
        
        # UI for Settings
        st.markdown("### 2. Configure Forecast")
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("**Target Level**")
            agg_mode = st.segmented_control("Level", ["Aggregate", "Product"], default="Aggregate")
            
            if agg_mode == "Product":
                product_list = df_raw[id_col].unique()
                selected_item = st.selectbox("Select Product", product_list)
            else:
                selected_item = "Total Sum"

        with c2:
            st.write("**Timeline**")
            interval = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"], index=1)
            horizon_val = st.number_input("Forecast Horizon (Steps)", min_value=1, value=12)

        with c3:
            st.write("**Model Logic**")
            technique = st.selectbox("Baseline Formula", ["Historical Avg", "Moving Avg", "Weighted Avg", "Exponential"])
            
            # Contextual settings for technique
            tech_params = {}
            if technique == "Moving Avg":
                tech_params['n'] = st.slider("Window Size", 2, 20, 7)
            elif technique == "Weighted Avg":
                tech_params['weights'] = [0.2, 0.3, 0.5] # Default
                st.caption("Using weights: 0.2, 0.3, 0.5 (Last 3 periods)")

        st.markdown('</div>', unsafe_allow_html=True)

        # Execution Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 1])
        with col_btn_2:
            run_forecast = st.button("🚀 Generate AI Forecast")

        # --- 5. LOGIC & RESULTS ---
        if run_forecast:
            # Data Processing
            df_long = df_raw.melt(id_vars=[id_col], value_vars=date_cols, var_name='Date', value_name='qty')
            df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
            df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
            df_long = df_long.dropna(subset=['Date']).sort_values('Date')

            if agg_mode == "Product":
                target_df = df_long[df_long[id_col] == selected_item].copy()
            else:
                target_df = df_long.groupby('Date')['qty'].sum().reset_index()

            # Resample
            res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
            target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

            # Baseline Calc
            history = target_df['qty'].tolist()
            if technique == "Moving Avg":
                base_val = np.mean(history[-tech_params.get('n', 7):])
            else:
                base_val = np.mean(history)

            # AI Model (XGBoost)
            target_df['month'] = target_df['Date'].dt.month
            target_df['diff'] = target_df['qty'] - base_val
            model = XGBRegressor(n_estimators=50)
            model.fit(target_df[['month']], target_df['diff'])

            # Future Prediction
            last_date = target_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=horizon_val+1, freq=res_map[interval])[1:]
            
            f_df = pd.DataFrame({'Date': future_dates, 'month': future_dates.month})
            ai_resid = model.predict(f_df[['month']])
            
            excel_fc = [round(base_val, 2)] * horizon_val
            ai_fc = [round(max(base_val + r, 0), 2) for r in ai_resid]

            # Visuals
            st.divider()
            st.markdown(f"### 📈 Forecast Analysis: {selected_item}")
            
            chart_col, data_col = st.columns([2, 1])
            
            with chart_col:
                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Actuals", line=dict(color='#4f46e5', width=3)))
                fig.add_trace(go.Scatter(x=future_dates, y=excel_fc, name="Baseline", line=dict(color='#94a3b8', dash='dot')))
                fig.add_trace(go.Scatter(x=future_dates, y=ai_fc, name="AI Prediction", line=dict(color='#10b981', width=4)))
                
                fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with data_col:
                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                res_df = pd.DataFrame({"Date": future_dates.strftime('%Y-%m-%d'), "Forecast": ai_fc})
                st.dataframe(res_df, use_container_width=True, height=330, hide_index=True)
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button("📥 Download Excel", data=output.getvalue(), file_name="forecast.xlsx")
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing data: {e}")

else:
    # Placeholder when no file is uploaded
    st.info("👋 Welcome! Please upload your order history file above to start the forecasting engine.")
    
    # Simple instructions cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Upload")
        st.caption("Drop your CSV or Excel file containing historical quantities.")
    with c2:
        st.markdown("#### 2. Configure")
        st.caption("Choose aggregation level, time intervals, and formula settings.")
    with c3:
        st.markdown("#### 3. AI Magic")
        st.caption("Our XGBoost model adjusts the baseline for seasonal patterns.")
