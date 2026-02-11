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

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #111827;
    }

    /* Remove Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Content Centering */
    .block-container {
        padding-top: 5rem;
        max-width: 1000px;
        margin: 0 auto;
    }

    /* Vertical Flow Design */
    .step-wrapper {
        position: relative;
        padding-left: 40px;
        margin-bottom: 50px;
        border-left: 2px solid #E5E7EB;
    }

    .step-dot {
        position: absolute;
        left: -9px;
        top: 0;
        width: 16px;
        height: 16px;
        background-color: #FFFFFF;
        border: 2px solid #4F46E5;
        border-radius: 50%;
    }

    .step-label {
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }

    .step-heading {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 20px;
    }

    /* Input Styling */
    .stSelectbox, .stRadio, .stNumberInput, .stTextInput {
        background-color: #F9FAFB;
        border-radius: 8px;
    }

    /* Premium Button */
    div.stButton > button {
        width: 100%;
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 18px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #4F46E5 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Result Insights Card */
    .insight-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 40px;
        border-radius: 24px;
        margin-top: 40px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div style="text-align: center; margin-bottom: 80px;">'
            '<h1 style="font-size: 3.5rem; font-weight: 800; letter-spacing: -2px; color: #4F46E5;">Agilo<span style="color:#111827;">Forecast</span></h1>'
            '<p style="color: #6B7280; font-size: 1.3rem;">AI-Driven Precision for Manufacturing Supply Chains</p>'
            '</div>', unsafe_allow_html=True)

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

# --- STEP 2: INTERVAL ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 02</div><div class="step-heading">Set Parameters</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    interval = st.selectbox("Forecast Interval", options=["Daily", "Weekly", "Monthly"], index=0)
with c2:
    horizon_label = st.selectbox("Forecast Horizon", ["Short (7 Days)", "Standard (15 Days)", "Medium (30 Days)", "Long (60 Days)"], index=1)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: TECHNIQUE ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 03</div><div class="step-heading">Baseline Baseline Technique</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    technique = st.selectbox("Statistical Method", ["Historical Average", "Weightage Average", "Moving Average", "Ramp Up Evenly", "Exponentially"])
with c4:
    tech_params = {}
    if technique == "Weightage Average":
        w_in = st.text_input("Manual Weight Ratios (Newest to Oldest)", "0.5, 0.3, 0.2")
        try: tech_params['weights'] = np.array([float(x.strip()) for x in w_in.split(',')])
        except: tech_params['weights'] = np.array([0.5, 0.3, 0.2])
    elif technique == "Moving Average":
        tech_params['n'] = st.number_input("Lookback Window", 2, 30, 7)
    elif technique == "Ramp Up Evenly":
        tech_params['ramp_factor'] = st.number_input("Growth Coefficient", 1.0, 2.0, 1.05)
    elif technique == "Exponentially":
        tech_params['alpha'] = st.slider("Smoothing Alpha", 0.01, 1.0, 0.3)
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 4: UPLOAD ---
st.markdown('<div class="step-wrapper"><div class="step-dot"></div>'
            '<div class="step-label">Step 04</div><div class="step-heading">Data Engine</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop Enterprise Excel/CSV Data", type=['xlsx', 'csv'])
st.markdown('</div>', unsafe_allow_html=True)

# --- CORE MATH LOGIC ---
def calculate_excel_baseline(demand, tech, params):
    if len(demand) == 0: return 0
    if tech == "Historical Average": return np.mean(demand)
    elif tech == "Moving Average":
        n = params.get('n', 7)
        return np.mean(demand[-n:]) if len(demand) >= n else np.mean(demand)
    elif tech == "Weightage Average":
        w = params.get('weights', np.array([0.5, 0.3, 0.2]))
        n = len(w)
        # We take the last n elements and multiply by weights (reversed to match newest -> oldest)
        if len(demand) >= n:
            return np.dot(demand[-n:], w[::-1]) / np.sum(w)
        return np.mean(demand)
    elif tech == "Ramp Up Evenly":
        d_slice = demand[-7:] if len(demand) >= 7 else demand
        weights = np.arange(1, len(d_slice) + 1)
        return np.dot(d_slice, weights) / weights.sum()
    elif tech == "Exponentially":
        alpha = params.get('alpha', 0.3)
        forecast = demand[0]
        for d in demand[1:]: forecast = alpha * d + (1 - alpha) * forecast
        return forecast
    return np.mean(demand)

# --- EXECUTION ENGINE ---
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        
        # COLUMN IDENTIFICATION
        model_col = raw.columns[0]   # Index 0: MODEL
        part_no_col = raw.columns[1] # Index 1: PART NO
        desc_col = raw.columns[2]    # Index 2: PART DESCRIPTION
        qty_veh_col = raw.columns[3] # Index 3: QTY/VEH
        
        # Everything after the first 4 columns are dates
        date_cols = [c for c in raw.columns[4:] if pd.to_datetime(c, errors='coerce', dayfirst=True) is not pd.NaT]
        
        # MELT INTO LONG FORMAT
        df_long = raw.melt(
            id_vars=[model_col, part_no_col, desc_col, qty_veh_col], 
            value_vars=date_cols, 
            var_name='RawDate', 
            value_name='qty'
        )
        df_long['Date'] = pd.to_datetime(df_long['RawDate'], dayfirst=True, format='mixed')
        df_long['qty'] = pd.to_numeric(df_long['qty'], errors='coerce').fillna(0)
        df_long = df_long.sort_values('Date').dropna(subset=['Date'])

        # FILTERING LOGIC
        if main_choice == "Aggregate Wise":
            target_df = df_long.groupby('Date')['qty'].sum().reset_index()
            item_name = "Global System Aggregate"
        else:
            if sub_choice == "Model Wise":
                selected_model = st.selectbox("🎯 Identity Model", df_long[model_col].unique())
                target_df = df_long[df_long[model_col] == selected_model].groupby('Date')['qty'].sum().reset_index()
                item_name = f"Model: {selected_model}"
            else:
                # PART NO WISE (YOUR REQUESTED CHANGE)
                selected_model = st.selectbox("🎯 Identity Model", df_long[model_col].unique())
                # Filter Part Nos based on the selected Model
                part_list = df_long[df_long[model_col] == selected_model][part_no_col].unique()
                selected_part = st.selectbox("📦 Identity Part No", part_list)
                
                target_df = df_long[
                    (df_long[model_col] == selected_model) & 
                    (df_long[part_no_col] == selected_part)
                ].copy()
                item_name = f"{selected_part} ({selected_model})"

        # RESAMPLING
        res_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        target_df = target_df.set_index('Date').resample(res_map[interval]).sum().reset_index()

        if st.button("EXECUTE PREDICTIVE ENGINE"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            
            # CALCULATE HISTORICAL TRENDS
            history = target_df['qty'].tolist()
            excel_base_scalar = calculate_excel_baseline(history, technique, tech_params)
            
            # AI LAYER (XGBOOST)
            target_df['month'] = target_df['Date'].dt.month
            target_df['dow'] = target_df['Date'].dt.dayofweek
            target_df['residual'] = target_df['qty'] - excel_base_scalar
            
            ai_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
            ai_model.fit(target_df[['month', 'dow']], target_df['residual'])
            
            # FUTURE DATES
            last_date = target_df['Date'].max()
            h_val = {"Short (7 Days)": 7, "Standard (15 Days)": 15, "Medium (30 Days)": 30, "Long (60 Days)": 60}[horizon_label]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h_val, freq=res_map[interval])
            
            f_df = pd.DataFrame({'Date': future_dates})
            f_df['month'], f_df['dow'] = f_df['Date'].dt.month, f_df['Date'].dt.dayofweek
            ai_residuals = ai_model.predict(f_df[['month', 'dow']])
            
            excel_calc_col, predicted_calc_col = [], []
            for i, res in enumerate(ai_residuals, 1):
                base = excel_base_scalar * (tech_params.get('ramp_factor', 1.05) ** i) if technique == "Ramp Up Evenly" else excel_base_scalar
                excel_calc_col.append(round(base, 2))
                predicted_calc_col.append(round(max(base + res, 0), 2))

            # --- 📈 TREND GRAPH (CURVY SPLINE) ---
            st.subheader(f"📊 Demand Forecast: {item_name}")
            fig = go.Figure()

            # Historical Actuals
            fig.add_trace(go.Scatter(
                x=target_df['Date'], y=target_df['qty'], name="Traded Demand",
                mode='lines+markers', line=dict(color="#1a8cff", width=3, shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#1a8cff", width=2))
            ))

            # Forecast Connections
            f_dates_conn = [target_df['Date'].iloc[-1]] + list(future_dates)
            f_excel_conn = [target_df['qty'].iloc[-1]] + list(excel_calc_col)
            f_pred_conn = [target_df['qty'].iloc[-1]] + list(predicted_calc_col)

            # Statistical Baseline
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_excel_conn, name="Statistical Baseline",
                mode='lines', line=dict(color="#94a3b8", width=1.5, dash='dot', shape='spline')
            ))

            # AI Forecast
            fig.add_trace(go.Scatter(
                x=f_dates_conn, y=f_pred_conn, name="AI Enhanced Forecast",
                mode='lines+markers', line=dict(color="#fbbf24", width=3, dash='dash', shape='spline'),
                marker=dict(size=6, color="white", line=dict(color="#fbbf24", width=2))
            ))

            fig.add_vline(x=last_date, line_width=2, line_color="#cbd5e1", line_dash="dash")
            fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(t=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # --- 📉 AI PATTERN CHART (WIGGLES) ---
            st.subheader("💡 AI Pattern Intelligence (Residuals)")
            st.info("The bars represent seasonal/cyclic variations the AI identified and added/subtracted from the math baseline.")
            fig_wig = go.Figure(go.Bar(
                x=future_dates, y=ai_residuals, 
                name="AI Pattern Adjust", marker_color="#4F46E5", marker_opacity=0.7
            ))
            fig_wig.update_layout(template="plotly_white", height=250, margin=dict(t=10))
            st.plotly_chart(fig_wig, use_container_width=True)

            # DATA TABLE
            st.markdown("#### Operational Demand Schedule")
            res_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'), 
                "AI Prediction": predicted_calc_col, 
                "Statistical Baseline": excel_calc_col
            })
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("📥 EXPORT INTELLIGENCE REPORT", output.getvalue(), f"Forecasting_Report_{item_name}.xlsx")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"System Error: {e}. Please ensure your file columns follow: Model | Part No | Description | Qty | Dates...")
