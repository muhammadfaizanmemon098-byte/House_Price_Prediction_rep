import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("house_price_model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #1f4037, #99f2c8);
}
.main {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.2);
}
h1 {
    color: #1f4037;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown("<h1>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.write("Enter the house feature values below:")

# ---------------- INPUTS ----------------
features = []
cols = st.columns(2)

for i in range(10):
    with cols[i % 2]:
        value = st.number_input(
            f"Feature {i+1}",
            min_value=0.0,
            step=0.1
        )
        features.append(value)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price"):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)[0]

    st.markdown(
        f"""
        <div style="
            background:#e7f8f1;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:20px;
            font-weight:bold;
            color:#1f4037;
        ">
        💰 Predicted House Price: {round(prediction, 2)}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:gray;'>Developed using Streamlit & Decision Tree</p>",
    unsafe_allow_html=True
)
