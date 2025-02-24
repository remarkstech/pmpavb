import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import traceback
import datetime  # Pastikan import datetime di awal

# ==========================
# 1. Load Model & Scaler (Dengan Caching)
# ==========================
@st.cache_resource
def load_model():
    """Load model hanya sekali untuk mencegah reload berulang."""
    model_path = os.path.abspath("dl_model.h5")  # Pastikan path absolut
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
st.title("Media Budget Prediction")

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
# One-hot encoding industri
industry_dict = {industry: 0 for industry in industries}
industry_dict[selected_industry] = 1

# Buat DataFrame dari input user
input_data = pd.DataFrame([{
    "impressions": impressions,
    "clicks": clicks,
    "leads": leads,
    "cpl": c
