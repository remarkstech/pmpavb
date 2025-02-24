import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# ==========================
# 1. Load Model & Scaler
# ==========================
# Path model & scaler
model_path = os.path.join(os.path.dirname(__file__), "dl_model.h5")
scaler_X_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler_y_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")  # Assuming both scalers are in the same file

# Load model and scaler
try:
    model = tf.keras.models.load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# ==========================
# 2. Streamlit Interface
# ==========================
st.title("Cost Estimation dengan Deep Learning")

# Input user
impressions = st.number_input("Impressions", min_value=0.0, format="%.0f")
clicks = st.number_input("Clicks", min_value=0.0, format="%.0f")
leads = st.number_input("Leads", min_value=0.0, format="%.0f")
cpl = st.number_input("CPL", min_value=0.0, format="%.0f")
cpc = st.number_input("CPC", min_value=0.0, format="%.0f")

# Dropdown untuk memilih industri
industries = ["AUTOMOTIVE", "BEAUTY", "EDUCATION", "FOOD MANUFACTURE", "LIFT DISTRIBUTOR", "PROPERTY"]
selected_industry = st.selectbox("Pilih Industri", industries)

# ==========================
# 3. Prepare Input Data
# ==========================
# Set nilai one-hot encoding industri
industry_dict = {industry: 0 for industry in industries}
industry_dict[selected_industry] = 1

# Buat DataFrame dari input user
input_data = pd.DataFrame([{
    "impressions": impressions,
    "clicks": clicks,
    "leads": leads,
    "cpl": cpl,
    "cpc": cpc,
    **{f"industry_{industry}": industry_dict[industry] for industry in industries}
}])

# Log transformation input
input_data_log = np.log1p(input_data)

# Scaling input
input_scaled = scaler_X.transform(input_data_log)

# ==========================
# 4. Model Prediction
# ==========================
if st.button("Calculate"):
    try:
        # Prediksi dengan model
        pred_scaled = model.predict(input_scaled)

        # Inverse transform hasil prediksi
        pred_log = scaler_y.inverse_transform(pred_scaled)
        predicted_cost = np.expm1(pred_log)[0, 0]  # Ubah ke angka asli

        # Tampilkan hasil prediksi
        st.success(f"Cost Estimation: IDR {predicted_cost:,.0f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
