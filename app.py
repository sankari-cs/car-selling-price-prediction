# app.py
import streamlit as st
import pickle
import pandas as pd

# Load artifacts
with open("model.pkl", "rb") as f: model = pickle.load(f)
with open("encoders.pkl", "rb") as f: encoders = pickle.load(f)
with open("feature_names.pkl", "rb") as f: feature_names = pickle.load(f)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Selling Price Predictor")
st.write("Enter the car details below to get an estimated selling price (in lakhs).")

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present/Showroom Price (₹ lakhs)", 0.0, 100.0, 5.0, 0.25)
    kms_driven    = st.number_input("Kilometers Driven", 0, 500000, 30000, 1000)
    owner         = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    car_age     = st.slider("Car Age (years)", 0, 30, 5)
    fuel_type   = st.selectbox("Fuel Type",   encoders["fuel_type"].classes_)
    seller_type = st.selectbox("Seller Type", encoders["seller_type"].classes_)
    transmission= st.selectbox("Transmission",encoders["transmission"].classes_)

if st.button("💰 Predict Selling Price", use_container_width=True):
    row = {
        "present_price": present_price,
        "kms_driven":    kms_driven,
        "fuel_type":     encoders["fuel_type"].transform([fuel_type])[0],
        "seller_type":   encoders["seller_type"].transform([seller_type])[0],
        "transmission":  encoders["transmission"].transform([transmission])[0],
        "owner":         owner,
        "car_age":       car_age,
    }
    X_in = pd.DataFrame([row])[feature_names]
    price = model.predict(X_in)[0]
    st.success(f"### Estimated Selling Price: ₹ {price:.2f} lakhs")
    st.caption("Note: This is an ML estimate based on historical data, not a guaranteed market price.")

st.divider()
st.caption("Built with Streamlit + scikit-learn | Car Selling Price Prediction Project")