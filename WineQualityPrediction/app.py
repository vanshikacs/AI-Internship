import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷"
)

st.title("🍷 Wine Quality Prediction")

model = joblib.load("wine_model.pkl")

fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.70)
citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.00)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.number_input("Density", 0.9900, 1.0100, 0.9978)
pH = st.number_input("pH", 2.0, 5.0, 3.51)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 0.0, 20.0, 9.4)

if st.button("Predict Quality"):

    sample = pd.DataFrame({
        "fixed acidity":[fixed_acidity],
        "volatile acidity":[volatile_acidity],
        "citric acid":[citric_acid],
        "residual sugar":[residual_sugar],
        "chlorides":[chlorides],
        "free sulfur dioxide":[free_sulfur_dioxide],
        "total sulfur dioxide":[total_sulfur_dioxide],
        "density":[density],
        "pH":[pH],
        "sulphates":[sulphates],
        "alcohol":[alcohol]
    })

    prediction = model.predict(sample)

    st.success(f"Predicted Wine Quality: {prediction[0]}")