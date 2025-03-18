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
    model_path = os.path.abspath("dl_model2.h5")  # Pastikan path absolut
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_scalers():
    """Load scaler_X dan scaler_y hanya sekali."""
    scaler_X_path = os.path.abspath("scaler_X.pkl")
    scaler_y_path = os.path.abspath("scaler_y.pkl")
    
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
# Tambahkan logo sebelum judul
st.image("logo_final-02.png", width=100)  # Pastikan file ada di lokasi yang benar
st.title("Media Plan Automation")

# Input user
impressions = st.number_input("Impressions", min_value=0.0, format="%.0f")
clicks = st.number_input("Clicks", min_value=0.0, format="%.0f")
leads = st.number_input("Leads", min_value=0.0, format="%.0f")
cpl = st.number_input("CPL", min_value=0.0, format="%.0f")
cpc = st.number_input("CPC", min_value=0.0, format="%.0f")

# Dropdown untuk memilih industri, dengan placeholder yang tidak bisa dipilih
industries = ["AUTOMOTIVE", "BEAUTY", "EDUCATION", "FOOD MANUFACTURE", "LIFT DISTRIBUTOR", "PROPERTY"]
selected_industry = st.selectbox("Industry Select", ["Select Industry"] + industries)  # Tambahkan "Select Industry" sebagai placeholder

# Dropdown untuk memilih campaign type, dengan placeholder yang tidak bisa dipilih
sources = ['DIRECT', 'FB', 'IG', 'SEM', 'DISC', 'PMAX']
selected_source = st.selectbox("Campaign Type", ["Campaign Type", "Others"] + sources)  # Tambahkan "Select Industry" sebagai placeholder

# ==========================
# 3. Prepare Input Data
# ==========================
# Cek apakah industri dipilih, jika belum beri peringatan
if selected_industry == "Select Industry":
    st.error("Please select a valid industry.")
else:
    # One-hot encoding industri
    industry_dict = {industry: 0 for industry in industries}  # Hanya encode industri yang valid
    industry_dict[selected_industry] = 1

# Cek apakah source dipilih
if selected_source == "Campaign Type":
    source_dict = {source: 0 for source in sources} 
else:
    source_dict = {source: 0 for source in sources}  # Default all sources to 0
    source_dict[selected_source] = 1

    # Buat DataFrame dari input user
    input_data = pd.DataFrame([{
        "impressions": impressions,
        "clicks": clicks,
        "leads": leads,
        "cpl": cpl,
        "cpc": cpc,
        **{f"source_{source}": source_dict[source] for source in sources},
        **{f"industry_{industry}": industry_dict[industry] for industry in industries}
    }])

    # ==========================
    # 4. Ensure Column Order (Matching Model Input Order)
    # ==========================
    expected_columns = [
        "impressions", "clicks", "leads", "cpl", "cpc",
        "source_DISC", "source_FB", "source_IG", "source_PMAX", "source_SEM",
        "industry_AUTOMOTIVE", "industry_EDUCATION", "industry_FOOD MANUFACTURE", 
        "industry_LIFT DISTRIBUTOR", "industry_PROPERTY"
    ]

    # Urutkan kolom sesuai dengan urutan yang diharapkan
    input_data = input_data[expected_columns]

    # **Pastikan scaler sudah di-fit sebelum transformasi**
    if not hasattr(scaler_X, "mean_") or not hasattr(scaler_y, "mean_"):
        st.error("Scaler belum di-fit dengan data. Pastikan scaler valid.")
        st.stop()

    # Log transformation input
    input_data_log = np.log1p(input_data)

    # Scaling input
    try:
        input_scaled = scaler_X.transform(input_data_log)
    except Exception as e:
        st.error(f"Error saat scaling input: {e}")
        st.stop()

    # ==========================
    # 5. Model Prediction
    # ==========================
    if st.button("Calculate"):
        try:
            with st.spinner("Predicting..."):
                # Prediksi dengan model
                pred_scaled = model.predict(input_scaled)

                # Inverse transform hasil prediksi
                pred_log = scaler_y.inverse_transform(pred_scaled)  # Balikkan scaling ke bentuk asli
                predicted_cost = np.expm1(pred_log)[0, 0]  # Kembalikan hasil dari log transformasi
                predicted_cost = predicted_cost * 1.05  # Tambahkan 5% pada hasil prediksi

            # Tampilkan hasil prediksi
            st.success(f"Cost Estimation: IDR {predicted_cost:,.0f}")

        except Exception as e:
            st.error("Terjadi kesalahan saat prediksi.")
            st.text(traceback.format_exc())  # Menampilkan log lengkap error

# ==========================
# Footer (Copyright & Remarks)
# ==========================
current_year = datetime.datetime.now().year  # Ambil tahun saat ini
st.markdown("---")  # Pembatas garis
st.markdown(f"© {current_year} Remarks Asia. All Rights Reserved.")
