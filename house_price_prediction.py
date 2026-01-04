import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

model = joblib.load("house_price_model.pkl")

st.markdown("""
<style>
.main {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
}
h1 {
    text-align: center;
    color: #1f4037;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.write("Enter feature values below:")

features = []
for i in range(1, 11):
    features.append(st.number_input(f"Feature {i}", value=0.0))

if st.button("Predict"):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)[0]
    st.success(f"💰 Predicted House Price: {round(prediction, 2)}")
