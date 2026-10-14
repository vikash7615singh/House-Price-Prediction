
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

st.title("House Price Prediction")

if not MODEL_PATH.exists():
    st.error("Model file not found. Train and save your model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.write("Enter the details of your house.")

area = st.number_input(
    "Area (square feet)", min_value=100,
    max_value=20000, value=1200
)

bedrooms = st.number_input(
    "Bedrooms", min_value=1, max_value=20, value=3
)

bathrooms = st.number_input(
    "Bathrooms", min_value=1, max_value=20, value=2
)

age = st.number_input(
    "House age (years)", min_value=0,
    max_value=200, value=5
)

if st.button("Predict Price"):
    house = pd.DataFrame({
        "area_sqft": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "house_age_years": [age]
    })

    price = model.predict(house)[0]

    st.success(f"Estimated price: ₹{price:.2f} lakh")

    st.caption(
        "Educational demonstration using invented sample data. "
        "Not a reliable real-estate valuation."
    )
