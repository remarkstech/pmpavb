import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "Model", "rf_model.joblib")
model_path = os.path.abspath(model_path)  # Ubah jadi path absolut

# Cek apakah file model benar-benar ada
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

# Load model
model = joblib.load(model_path)

# Judul aplikasi
st.title("CPL Prediction")

# Input user
impressions = st.number_input("Impressions", min_value=0.0, format="%.0f")
clicks = st.number_input("Clicks", min_value=0.0, format="%.0f")
leads = st.number_input("Leads", min_value=0.0, format="%.0f")
cpl = st.number_input("CPL", min_value=0.0, format="%.0f")
cpc = st.number_input("CPC", min_value=0.0, format="%.0f")
cpm = st.number_input("CPM", min_value=0.0, format="%.0f")

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Buat DataFrame dari input user
        input_data = pd.DataFrame([{ 
            "impressions": impressions, 
            "clicks": clicks, 
            "leads": leads, 
            "cpl": cpl, 
            "cpc": cpc, 
            "cpm": cpm 
        }])

        # Lakukan prediksi
        result = np.ceil(model.predict(input_data) * 1.2).astype(np.int64)

        # Tampilkan hasil prediksi
        st.success(f"Hasil Prediksi: {result[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
