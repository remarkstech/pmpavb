import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import traceback
import datetime

# ==========================
# 1. Load Model & Scaler
# ==========================
@st.cache_resource
def load_model():
    try:
        model_path = os.path.abspath("dl_model3.h5")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_scalers():
    try:
        scaler_X = joblib.load("scaler_X1.pkl")
        scaler_y = joblib.load("scaler_y1.pkl")
        return scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading scalers: {e}")
        st.stop()

# ==========================
# Start Streamlit App
# ==========================
st.image("logo_final-02.png", width=100)
st.title("Media Plan Automation")

# Model Switch
model_type = st.radio("Select Model Type", ["Model 1 (Prediction Model)", "Model 2 (Manual Calculation)"])

# ==========================
# MODEL 1 - Neural Network
# ==========================
if model_type == "Model 1 (Prediction Model)":
    with st.spinner("Loading model & scalers..."):
        model = load_model()
        scaler_X, scaler_y = load_scalers()

    st.markdown("""
    **Instructions:**
    1. Enter the required metrics (Impressions, Clicks, Leads, CPL, CPC).
    2. Select the Industry and Campaign Type.
    3. Click 'Calculate Cost' to get the cost estimation.
    """)

    margin = st.number_input("Margin (%)", min_value=0.0, format="%.0f", value=0.0)
    impressions = st.number_input("Impressions", min_value=0.0, format="%.0f", value=0.0)
    clicks = st.number_input("Clicks", min_value=0.0, format="%.0f", value=0.0)
    leads = st.number_input("Leads", min_value=0.0, format="%.0f", value=0.0)
    cpc = st.number_input("CPC", min_value=0.0, format="%.2f", value=0.0)
    cpl = st.number_input("CPL", min_value=0.0, format="%.2f", value=0.0)

    sources = ['Select Campaign Type', 'DIRECT', 'FB', 'IG', 'SEM', 'DISC', 'PMAX']
    industries = ['Select Industry Type', "AUTOMOTIVE", "EDUCATION", "FOOD MANUFACTURE", "LIFT DISTRIBUTOR", "PROPERTY"]
    selected_source = st.selectbox("Campaign Type", sources, index=0)
    selected_industry = st.selectbox("Industry", industries, index=0)

    industry_dict = {industry: 0 for industry in industries[1:]}
    if selected_industry != "Select Industry Type":
        industry_dict[selected_industry] = 1

    source_dict = {source: 0 for source in sources[1:]}
    if selected_source != "Select Campaign Type":
        source_dict[selected_source] = 1

    input_data = pd.DataFrame([{
        "impressions": impressions,
        "clicks": clicks,
        "leads": leads,
        "cpl": cpl,
        "cpc": cpc,
        **{f"source_{s}": source_dict[s] for s in sources[1:]},
        **{f"industry_{i}": industry_dict[i] for i in industries[1:]}
    }])

    expected_columns = [
        "impressions", "clicks", "leads", "cpl", "cpc",
        "source_DISC", "source_FB", "source_IG", "source_PMAX", "source_SEM",
        "industry_AUTOMOTIVE", "industry_EDUCATION", "industry_FOOD MANUFACTURE",
        "industry_LIFT DISTRIBUTOR", "industry_PROPERTY"
    ]
    input_data = input_data[expected_columns]
    input_scaled = scaler_X.transform(np.log1p(input_data))

    if st.button("Calculate Cost"):
        with st.spinner("Predicting..."):
            try:
                pred_scaled = model.predict(input_scaled)
                pred_log = scaler_y.inverse_transform(pred_scaled)
                predicted_cost = np.expm1(pred_log)

                final_cost = predicted_cost * (1 + (margin / 100)) if margin > 0 else predicted_cost
                label = "**Cost Estimation with Margin:**" if margin > 0 else "**Cost Estimation:**"
                st.success(f"{label} IDR {final_cost[0][0]:,.0f}")
            except Exception as e:
                st.error("An error occurred during prediction.")
                st.text(traceback.format_exc())

# ==========================
# MODEL 2 - Manual Calculation
# ==========================
elif model_type == "Model 2 (Manual Calculation)":
    import io
    import base64

    st.markdown("### Manual Cost Estimation")

    total_budget = st.number_input("💰 Total Budget (IDR)", min_value=0.0, format="%.0f", value=0.0)
    aov = st.number_input("🛒 Average Order Value / AOV (IDR)", min_value=0.0, format="%.0f", value=0.0)

    st.markdown("### Platform Inputs")
    platforms = ["Meta", "TikTok", "Google"]
    model2_data = {}

    for platform in platforms:
        with st.expander(f"{platform}"):
            reset = st.button(f"🔄 Reset {platform}", key=f"reset_{platform}")
            if reset:
                st.session_state[f"{platform}_inv"] = 0.0
                st.session_state[f"{platform}_ctr"] = 0.0
                st.session_state[f"{platform}_cpc"] = 0.0
                st.session_state[f"{platform}_cr"] = 0.0

            inv_percent = st.number_input(
                f"{platform} - Investment (%)", min_value=0.0, max_value=100.0,
                value=st.session_state.get(f"{platform}_inv", 0.0), format="%.2f", key=f"{platform}_inv"
            )
            ctr = st.number_input(
                f"{platform} - CTR (%)", min_value=0.0,
                value=st.session_state.get(f"{platform}_ctr", 0.0), format="%.2f", key=f"{platform}_ctr"
            )
            cpc = st.number_input(
                f"{platform} - CPC (IDR)", min_value=0.0,
                value=st.session_state.get(f"{platform}_cpc", 0.0), format="%.2f", key=f"{platform}_cpc"
            )
            cr = st.number_input(
                f"{platform} - CR (%)", min_value=0.0,
                value=st.session_state.get(f"{platform}_cr", 0.0), format="%.2f", key=f"{platform}_cr"
            )

            model2_data[platform] = {
                "inv_percent": inv_percent,
                "ctr": ctr,
                "cpc": cpc,
                "cr": cr
            }

    if st.button("🧮 Calculate Model 2 Result (Manual)"):
        st.subheader("📊 Model 2 Results (Using Formulas)")
        result_data = []

        for platform, data in model2_data.items():
            try:
                investment = total_budget * (data["inv_percent"] / 100)
                clicks = investment / data["cpc"] if data["cpc"] > 0 else 0
                impressions = clicks / (data["ctr"] / 100) if data["ctr"] > 0 else 0
                cpm = (investment / impressions) * 1000 if impressions > 0 else 0
                orders = clicks * (data["cr"] / 100)
                sales = orders * aov
                cps = investment / orders if orders > 0 else 0
                roas = sales / investment if investment > 0 else 0

                result_data.append({
                    "Platform": platform,
                    "Investment (IDR)": investment,
                    "CPC (IDR)": data["cpc"],
                    "CTR (%)": data["ctr"],
                    "CR (%)": data["cr"],
                    "Clicks": clicks,
                    "Impressions": impressions,
                    "CPM (IDR)": cpm,
                    "Orders": orders,
                    "Sales (IDR)": sales,
                    "CPS (IDR)": cps,
                    "ROAS": roas
                })
            except Exception as e:
                st.warning(f"Calculation error for {platform}: {e}")

        st.markdown(f"#### 💰 Total Budget: IDR {total_budget:,.0f}")
        st.markdown(f"#### 🛒 AOV: IDR {aov:,.0f}")

        result_df = pd.DataFrame(result_data)

        # Format tampil di tabel
        formatted_df = result_df.copy()
        formatted_df["Investment (IDR)"] = formatted_df["Investment (IDR)"].apply(lambda x: f"{x:,.0f}")
        formatted_df["CPC (IDR)"] = formatted_df["CPC (IDR)"].apply(lambda x: f"{x:,.2f}")
        formatted_df["CTR (%)"] = formatted_df["CTR (%)"].apply(lambda x: f"{x:.2f}")
        formatted_df["CR (%)"] = formatted_df["CR (%)"].apply(lambda x: f"{x:.2f}")
        formatted_df["Clicks"] = formatted_df["Clicks"].apply(lambda x: f"{x:,.0f}")
        formatted_df["Impressions"] = formatted_df["Impressions"].apply(lambda x: f"{x:,.0f}")
        formatted_df["CPM (IDR)"] = formatted_df["CPM (IDR)"].apply(lambda x: f"{x:,.2f}")
        formatted_df["Orders"] = formatted_df["Orders"].apply(lambda x: f"{x:,.0f}")
        formatted_df["Sales (IDR)"] = formatted_df["Sales (IDR)"].apply(lambda x: f"{x:,.0f}")
        formatted_df["CPS (IDR)"] = formatted_df["CPS (IDR)"].apply(lambda x: f"{x:,.2f}")
        formatted_df["ROAS"] = formatted_df["ROAS"].apply(lambda x: f"{x:.2f}x")

        st.dataframe(formatted_df, use_container_width=True, hide_index=True)

        # Download Excel
        towrite = io.BytesIO()
        result_df.to_excel(towrite, index=False, sheet_name="Model2 Result")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="model2_result.xlsx">📥 Download as Excel</a>'
        st.markdown(href, unsafe_allow_html=True)


# ==========================
# Footer
# ==========================
current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(f"<div style='text-align: center;'>© {current_year} Remarks Asia. All Rights Reserved.</div>", unsafe_allow_html=True)
