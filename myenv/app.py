import streamlit as st
import os
import numpy as np
import pandas as pd
from src.insurance.pipeline.prediction import PredictionPipeline

st.title("Insurance Expense Prediction")

# Button to trigger model training
if st.button("Train Model"):
    os.system("python main.py")  # Keep training separate from prediction
    st.success("Training Complete!")

# Form to take input data for prediction
with st.form("prediction_form"):
    st.header("Enter Insurance Details:")

    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])

    # Submit button for the form
    submitted = st.form_submit_button("Predict")

    if submitted:  # Only make predictions when form submitted to avoid unnecessary calls
        try:
            # Preprocess the input data as per model requirements
            sex = 1 if sex == "female" else 0
            smoker = 1 if smoker == "yes" else 0
            region_northeast = 1 if region == "northeast" else 0
            region_southeast = 1 if region == "southeast" else 0
            region_southwest = 1 if region == "southwest" else 0
            region_northwest = 1 if region == "northwest" else 0

            # Prepare input data as a pandas DataFrame with the correct columns
            input_data = {
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
              # Expenses are not used for prediction, set to 0
                'region_northwest': [region_northwest],
                'region_southeast': [region_southeast],
                'region_southwest': [region_southwest]
            }

            # Create DataFrame with the exact column names used during training
            input_df = pd.DataFrame(input_data)

            # Assuming you have a PredictionPipeline class that can make predictions
            obj = PredictionPipeline()
            prediction = obj.predict(input_df)

            if prediction is None:
                st.error("Prediction failed. Check your model and inputs.")
            else:
                st.success(f"Predicted Insurance Expenses: ${prediction[0]:.2f}")  # Show predicted expenses
        except Exception as e:
            st.error(f"An error occurred: {e}")
