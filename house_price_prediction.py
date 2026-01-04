import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load model
model = joblib.load("house_price_model.pkl")

# Title
st.markdown(
    "<h1 style='text-align:center;color:#1f4037;'>🏠 House Price Prediction</h1>",
    unsafe_allow_html=True
)

st.write("Enter the feature values below:")

# Input fields (10 features)
features = []
for i in range(1, 11):
    value = st.number_input(f"Feature {i}", value=0.0)
    features.append(value)

# Predict button
if st.button("Predict Price"):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)[0]

    st.success(f"💰 Predicted House Price: {round(prediction, 2)}")
