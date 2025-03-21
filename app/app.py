import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import traceback
import datetime

# ==========================
# 1. Load Model & Scaler (Dengan Caching)
# ==========================
@st.cache_resource
def load_model():
    """Load model hanya sekali untuk mencegah reload berulang."""
    model_path = os.path.abspath("dl_model3.h5")  
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_scalers():
    """Load scaler_X dan scaler_y hanya sekali."""
    scaler_X_path = os.path.abspath("scaler_X1.pkl")
    scaler_y_path = os.path.abspath("scaler_y1.pkl")
    
    try:
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        return scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

# Load model & scalers
with st.spinner("Loading model & scalers..."):
    model = load_model()
    scaler_X, scaler_y = load_scalers()

# ==========================
# 2. Streamlit Interface
# ==========================
st.image("logo_final-02.png", width=100)
st.title("Media Plan Automation")

# Pilihan Mode: Prediksi Cost atau Cari Fitur
# margin = st.number_input("Margin", min_value=1.0, format="%.0f")

    # Input user untuk prediksi cost
margin = st.number_input("Margin", min_value=1.0, format="%.0f")
    impressions = st.number_input("Impressions", min_value=0.0, format="%.0f")
    clicks = st.number_input("Clicks", min_value=0.0, format="%.0f")
    leads = st.number_input("Leads", min_value=0.0, format="%.0f")
    cpl = st.number_input("CPL", min_value=0.0, format="%.2f")
    cpc = st.number_input("CPC", min_value=0.0, format="%.2f")

    industries = ["AUTOMOTIVE", "EDUCATION", "FOOD MANUFACTURE", "LIFT DISTRIBUTOR", "PROPERTY"]
    selected_industry = st.selectbox("Industry", industries)

    sources = ['DIRECT', 'FB', 'IG', 'SEM', 'DISC', 'PMAX']
    selected_source = st.selectbox("Campaign Type", sources)

    # One-hot encoding industri & source
    industry_dict = {industry: 0 for industry in industries}
    industry_dict[selected_industry] = 1

    source_dict = {source: 0 for source in sources}
    source_dict[selected_source] = 1

    # Buat DataFrame
    input_data = pd.DataFrame([{
        "impressions": impressions,
        "clicks": clicks,
        "leads": leads,
        "cpl": cpl,
        "cpc": cpc,
        **{f"source_{source}": source_dict[source] for source in sources},
        **{f"industry_{industry}": industry_dict[industry] for industry in industries}
    }])

    # Urutkan sesuai model
    expected_columns = [
        "impressions", "clicks", "leads", "cpl", "cpc",
        "source_DISC", "source_FB", "source_IG", "source_PMAX", "source_SEM",
        "industry_AUTOMOTIVE", "industry_EDUCATION", "industry_FOOD MANUFACTURE", 
        "industry_LIFT DISTRIBUTOR", "industry_PROPERTY"
    ]
    input_data = input_data[expected_columns]

    # Transformasi input
    input_data_log = np.log1p(input_data)
    input_scaled = scaler_X.transform(input_data_log)

    # Prediksi cost
    if st.button("Calculate Cost"):
        try:
            with st.spinner("Predicting..."):
                pred_scaled = model.predict(input_scaled)
                pred_log = scaler_y.inverse_transform(pred_scaled)
                predicted_cost = np.expm1(pred_log)
                predicted_cost = predicted_cost * 1.05
                predicted_cost2 = predicted_cost * (1 + (margin/100))
                
                st.success(f"Cost Estimation: IDR {predicted_cost:,.0f}")
                st.success(f"Cost Estimation with margin : IDR {predicted_cost2:,.0f}")
        except Exception as e:
            st.error("Terjadi kesalahan saat prediksi.")
            st.text(traceback.format_exc())
            
# ==========================
# Footer
# ==========================
current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(f"© {current_year} Remarks Asia. All Rights Reserved.")
