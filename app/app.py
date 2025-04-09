# ==========================
# IMPORT LIBRARY
# ==========================
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import traceback
import datetime

# ==========================
# LOAD MODEL & SCALERS (CACHED)
# ==========================
@st.cache_resource
def load_model():
    model_path = os.path.abspath("dl_model3.h5")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_scalers():
    scaler_X_path = os.path.abspath("scaler_X1.pkl")
    scaler_y_path = os.path.abspath("scaler_y1.pkl")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    return scaler_X, scaler_y

model = load_model()
scaler_X, scaler_y = load_scalers()

# ==========================
# STREAMLIT UI
# ==========================
st.image("logo_final-02.png", width=100)
st.title("Media Plan Automation")

model_choice = st.radio("Pilih Mode Prediksi:", ("Model 1 - AI", "Model 2 - Manual"))

# ==========================
# MODEL 1
# ==========================
if model_choice == "Model 1 - AI":
    st.markdown("### 🔮 Model 1 - AI Prediction")
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
        **{f"source_{source}": source_dict[source] for source in sources[1:]},
        **{f"industry_{industry}": industry_dict[industry] for industry in industries[1:]}
    }])

    expected_columns = [
        "impressions", "clicks", "leads", "cpl", "cpc",
        "source_DISC", "source_FB", "source_IG", "source_PMAX", "source_SEM",
        "industry_AUTOMOTIVE", "industry_EDUCATION", "industry_FOOD MANUFACTURE", 
        "industry_LIFT DISTRIBUTOR", "industry_PROPERTY"
    ]
    input_data = input_data[expected_columns]

    input_data_log = np.log1p(input_data)
    input_scaled = scaler_X.transform(input_data_log)

    if st.button("Calculate Cost (Model 1 - AI)"):
        try:
            pred_scaled = model.predict(input_scaled)
            pred_log = scaler_y.inverse_transform(pred_scaled)
            predicted_cost = np.expm1(pred_log)
            predicted_cost2 = predicted_cost * (1 + (margin / 100)) if margin > 0 else predicted_cost

            st.success(f"**Cost Estimation:** IDR {predicted_cost2[0][0]:,.0f}")
        except Exception as e:
            st.error("Error saat prediksi.")
            st.text(traceback.format_exc())

# ==========================
# MODEL 2
# ==========================
elif model_choice == "Model 2 - Manual":
    st.markdown("### 🧮 Model 2 - Manual Formula (per platform)")

    with st.form("model2_form"):
        st.write("Masukkan nilai untuk masing-masing platform:")
        model2_data = {}

        for platform in ["Meta", "TikTok", "Google"]:
            st.subheader(platform)
            inv = st.number_input(f"Investment ({platform})", min_value=0.0, value=0.0, format="%.0f", key=f"{platform}_inv")
            cpc_val = st.number_input(f"CPC ({platform})", min_value=0.01, value=1000.0, format="%.2f", key=f"{platform}_cpc")
            ctr_val = st.number_input(f"CTR (%) ({platform})", min_value=0.01, value=1.0, format="%.2f", key=f"{platform}_ctr")
            model2_data[platform] = {"inv": inv, "cpc": cpc_val, "ctr": ctr_val}

        submitted = st.form_submit_button("Calculate Model 2 Result (Manual)")

    if submitted:
        result_data = []
        total_inv = 0
        total_clicks = 0
        total_impressions = 0

        for platform, data in model2_data.items():
            try:
                clicks = data["inv"] / data["cpc"] if data["cpc"] > 0 else 0
                impressions = clicks / (data["ctr"] / 100) if data["ctr"] > 0 else 0
                cpm = (data["inv"] / impressions) * 1000 if impressions > 0 else 0

                total_inv += data["inv"]
                total_clicks += clicks
                total_impressions += impressions

                result_data.append({
                    "Platform": platform,
                    "Investment (IDR)": f"{data['inv']:,.0f}",
                    "CPC (IDR)": f"{data['cpc']:,.2f}",
                    "CTR (%)": f"{data['ctr']:.2f}",
                    "Clicks": f"{clicks:,.0f}",
                    "Impressions": f"{impressions:,.0f}",
                    "CPM (IDR)": f"{cpm:,.2f}"
                })
            except Exception as e:
                st.warning(f"Error menghitung {platform}: {e}")

        result_df = pd.DataFrame(result_data)
        st.dataframe(result_df, use_container_width=True)

        st.markdown("### 📊 Total Summary")
        st.markdown(f"""
        - **Total Investment:** IDR {total_inv:,.0f}  
        - **Total Clicks:** {total_clicks:,.0f}  
        - **Total Impressions:** {total_impressions:,.0f}
        """)

# ==========================
# FOOTER
# ==========================
current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(f"<div style='text-align: center;'>© {current_year} Remarks Asia. All Rights Reserved.</div>", unsafe_allow_html=True)
