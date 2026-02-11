import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import io

# --- 1. PREMIUM ENTERPRISE UI CONFIG ---
st.set_page_config(page_title="AI Supply Chain | Precision", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for a "SaaS Product" look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #FFFFFF; color: #111827; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 2rem; max-width: 1100px; margin: 0 auto; }
    .step-wrapper { position: relative; padding-left: 40px; margin-bottom: 30px; border-left: 2px solid #E5E7EB; }
    .step-dot { position: absolute; left: -9px; top: 0; width: 16px; height: 16px; background-color: #FFFFFF; border: 2px solid #4F46E5; border-radius: 50%; }
    .step-label { color: #4F46E5; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .step-heading { font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 20px; }
    div.stButton > button { width: 100%; background-color: #111827 !important; color: #FFFFFF !important; border: none !important; padding: 15px !important; font-size: 1rem !important; font-weight: 600 !important; border-radius: 10px !important; }
    .insight-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 30px; border-radius: 20px; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div style="text-align: center; margin-bottom: 40px;">'
            '<h1 style="font-size: 2.5rem; font-weight: 800; letter-spacing: -2px; color: #4F46E5;">Agilo<span style="color:#111827;">Forecast</span></h1>'
            '<p style="color: #6B7280; font-size: 1.1rem;">Precision Planning for Multi-Product Inventories</p>'
            '</div>', unsafe_allow_html=True)

# --- STEP 1: RESOLUTION ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 01</div><div class="step-heading">Forecasting Scope</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    main_choice = st.radio("Selection Path", ["Aggregate Wise (Total Factory)", "Product Wise"], horizontal=True)
with col2:
    if main_choice == "Product Wise":
        sub_choice = st.radio("Resolution Level", ["Model Wise", "Part No Wise"], horizontal=True)
    else:
        sub_choice = None
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: PARAMS ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 02</div><div class="step-heading">Set Parameters</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    interval = st.selectbox("Frequency of Data", options=["Daily", "Weekly", "Monthly"], index=0)
with c2:
    horizon_val = st.number_input("Forecast Horizon Length", min_value=1, value=15)
    horizon_unit = st.selectbox("Horizon Unit", ["Days", "Weeks", "Months"])
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUE ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 03</div><div class="step-heading">Forecast Technique</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    technique = st.selectbox("Baseline Algorithm", ["Historical Average", "Moving Average", "Exponentially"])
with c4:
    tech_params = {}
    if technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback Window", 2, 30, 7)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 04</div><div class="step-heading">Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV/Excel (Must contain MODEL, PART NO, PART DESCRIPTION, QTY/VEH columns)", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE LOGIC ---
def calculate_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

if uploaded_file:
    try:
        # Load Data
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        
        # Identify Columns based on your provided image structure
        meta_cols = ['MODEL', 'PART NO', 'PART DESCRIPTION', 'QTY/VEH']
        # Find date columns (everything that isn't metadata)
        date_cols = [c for c in raw.columns if c not in meta_cols]
        
        # Melt data into long format
        df_long = raw.melt(id_vars=meta_cols, value_vars=date_cols, var_name='RawDate', value_name='qty')
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, errors='coerce')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.dropna(subset=['Date']).sort_values('Date')

        # Hierarchical Selection UI
        target_df = pd.DataFrame()
        item_label = ""

        if main_choice == "Aggregate Wise (Total Factory)":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_label = "Total Factory Demand"
        
        elif sub_choice == "Model Wise":
            sel_model = st.selectbox("🎯 Select Model", df_long['MODEL'].unique())
            target_df = df_long[df_long['MODEL'] == sel_model].groupby('Date')['qty'].sum().reset_index()
            item_label = f"Model: {sel_model}"

        elif sub_choice == "Part No Wise":
            sel_model = st.selectbox("🎯 Select Model", df_long['MODEL'].unique())
            # Filter parts based on selected model
            available_parts = df_long[df_long['MODEL'] == sel_model]['PART NO'].unique()
            sel_part = st.selectbox("📦 Select Part Number", available_parts)
            
            # Extract info for display
            part_info = df_long[(df_long['MODEL'] == sel_model) & (df_long['PART NO'] == sel_part)].iloc[0]
            st.info(f"**Part Description:** {part_info['PART DESCRIPTION']} | **Qty per Veh:** {part_info['QTY/VEH']}")
            
            target_df = df_long[(df_long['MODEL'] == sel_model) & (df_long['PART NO'] == sel_part)].copy()
            item_label = f"Part: {sel_part} ({sel_model})"

        if not target_df.empty:
            # Resample based on interval
            res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
            target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

            if st.button("GENERATE AI FORECAST"):
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                
                # Calculations
                history = target_df['qty'].tolist()
                base_val = calculate_baseline(history, technique, tech_params)
                
                # Simple XGBoost Pattern Recognition
                target_df['month'] = target_df['Date'].dt.month
                target_df['dow'] = target_df['Date'].dt.dayofweek
                target_df['diff'] = target_df['qty'] - base_val
                
                model = XGBRegressor(n_estimators=50)
                model.fit(target_df[['month', 'dow']], target_df['diff'])
                
                # Create Future Dates
                last_date = target_df['Date'].max()
                freq_map = {"Days": "D", "Weeks": "W", "Months": "M"}
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_val, freq=freq_map[horizon_unit])
                
                f_df = pd.DataFrame({'Date': future_dates})
                f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
                ai_resid = model.predict(f_df[['month', 'dow']])
                
                # Final Forecasts
                preds = [round(max(base_val + r, 0), 2) for r in ai_resid]
                base_line = [round(base_val, 2)] * len(preds)

                # --- PLOT ---
                fig = go.Figure()
                # Historical
                fig.add_trace(go.Scatter(x=target_df['Date'], y=target_df['qty'], name="Actual Demand", line=dict(color="#1a8cff", width=3)))
                # Forecast
                fig.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Predicted", line=dict(color="#ffcc00", width=3, dash='dash')))
                
                fig.update_layout(title=f"Demand Projection: {item_label}", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # Data Table & Export
                res_df = pd.DataFrame({"Date": future_dates.strftime('%d-%m-%Y'), "Forecast Qty": preds})
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button("📥 DOWNLOAD REPORT", output.getvalue(), f"Forecast_{sel_part if 'sel_part' in locals() else 'Aggregate'}.xlsx")
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error parsing file: {e}. Ensure columns 'MODEL', 'PART NO', 'PART DESCRIPTION', and 'QTY/VEH' exist.")
