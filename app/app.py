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
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

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

# Instructions
st.markdown("""
    **Instructions:**
    1. Enter the required metrics (Impressions, Clicks, Leads, CPL, CPC).
    2. Select the Industry and Campaign Type.
    3. Click 'Calculate Cost' to get the cost estimation.
""")

# Inputs
margin = st.number_input("Margin", min_value=0.0, format="%.0f")
impressions = st.number_input("Impressions", min_value=0.0, format="%.0f")
clicks = st.number_input("Clicks", min_value=0.0, format="%.0f")
leads = st.number_input("Leads", min_value=0.0, format="%.0f")
cpl = st.number_input("CPL", min_value=0.0, format="%.2f")
cpc = st.number_input("CPC", min_value=0.0, format="%.2f")

# Add default options for Campaign Type and Industry
sources = ['Select Campaign Type', 'DIRECT', 'FB', 'IG', 'SEM', 'DISC', 'PMAX']
industries = ['Select Industry Type', "AUTOMOTIVE", "EDUCATION", "FOOD MANUFACTURE", "LIFT DISTRIBUTOR", "PROPERTY"]

selected_source = st.selectbox("Campaign Type", sources)
selected_industry = st.selectbox("Industry", industries)

# One-hot encoding industri & source
industry_dict = {industry: 0 for industry in industries[1:]}  # Exclude default option
if selected_industry != "Select Industry Type":
    industry_dict[selected_industry] = 1

source_dict = {source: 0 for source in sources[1:]}  # Exclude default option
if selected_source != "Select Campaign Type":
    source_dict[selected_source] = 1

# Buat DataFrame
input_data = pd.DataFrame([{
    "impressions": impressions,
    "clicks": clicks,
    "leads": leads,
    "cpl": cpl,
    "cpc": cpc,
    **{f"source_{source}": source_dict[source] for source in sources[1:]},  # Exclude default option
    **{f"industry_{industry}": industry_dict[industry] for industry in industries[1:]}  # Exclude default option
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
    with st.spinner("Predicting..."):
        # try:
        #     # Validasi input
        #     if selected_source == "Select Campaign Type":
        #         st.error("Please select a valid Campaign Type.")
        #         st.stop()
        #     if selected_industry == "Select Industry Type":
        #         st.error("Please select a valid Industry.")
        #         st.stop()

            # Make prediction
            pred_scaled = model.predict(input_scaled)
            pred_log = scaler_y.inverse_transform(pred_scaled)
            predicted_cost = np.expm1(pred_log)
            predicted_cost = predicted_cost * 1.05

            # Handle margin calculation
            if margin == 0:
                predicted_cost2 = predicted_cost
            else:
                predicted_cost2 = predicted_cost * (1 + (margin / 100))
            
            # Display results
            st.success(f"**Cost Estimation:** IDR {predicted_cost[0][0]:,.0f}")
            st.success(f"**Cost Estimation with Margin:** IDR {predicted_cost2[0][0]:,.0f}")
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.text(traceback.format_exc())

# Reset Button
if st.button("Reset Inputs"):
    st.experimental_rerun()

# ==========================
# Footer
# ==========================
current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(f"<div style='text-align: center;'>© {current_year} Remarks Asia. All Rights Reserved.</div>", unsafe_allow_html=True)
