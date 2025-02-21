import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Menambahkan CSS kustom
st.markdown("""
    <style>
    .stApp {
        background-color: #000080;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    .stNumberInput input {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Menambahkan path model
model_path = os.path.join(os.path.dirname(__file__), "Model", "rf_model.joblib")
model_path = os.path.abspath(model_path)  # Ubah jadi path absolut

# Cek apakah file model benar-benar ada
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

# Load model
model = joblib.load(model_path)

# Judul aplikasi
st.title("Remarks Cost Estimation")

# Input user
impressions = st.number_input("Impressions", min_value=0.0, format="%.0f")
clicks = st.number_input("Clicks", min_value=0.0, format="%.0f")
leads = st.number_input("Leads", min_value=0.0, format="%.0f")
cpl = st.number_input("CPL", min_value=0.0, format="%.0f")
cpc = st.number_input("CPC", min_value=0.0, format="%.0f")

# Dropdown untuk memilih industri
industries = ["AUTOMOTIVE", "BEAUTY", "EDUCATION", "FOOD MANUFACTURE", "LIFT DISTRIBUTOR", "PROPERTY"]
selected_industry = st.selectbox("Pilih Industri", industries)

# Menentukan kolom industri yang dipilih
industry_columns = [f"industry_{industry}" for industry in industries]

# Menangani pemilihan industri dan set nilai kolom industri
industry_dict = {industry: False for industry in industries}
industry_dict[selected_industry] = True

# Tombol prediksi
if st.button("Calculate"):
    try:
        # Buat DataFrame dari input user
        input_data = pd.DataFrame([{
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "cpl": cpl,
            "cpc": cpc,
            "industry_AUTOMOTIVE": industry_dict["AUTOMOTIVE"],
            "industry_BEAUTY": industry_dict["BEAUTY"],
            "industry_EDUCATION": industry_dict["EDUCATION"],
            "industry_FOOD MANUFACTURE": industry_dict["FOOD MANUFACTURE"],
            "industry_LIFT DISTRIBUTOR": industry_dict["LIFT DISTRIBUTOR"],
            "industry_PROPERTY": industry_dict["PROPERTY"]
        }])

        # Lakukan prediksi
        result = np.ceil(model.predict(input_data)).astype(np.int64)*1.35)

        # Tampilkan hasil prediksi dengan pemisah ribuan dan simbol IDR
        st.success(f"Cost Estimation: IDR {result[0]:,}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
