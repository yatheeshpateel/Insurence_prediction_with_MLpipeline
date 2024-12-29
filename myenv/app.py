import streamlit as st
import pandas as pd
import os
from src.insurance.pipeline.prediction import PredictionPipeline

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)  # Convert height to meters

# Indian insurance provider names and cost ranges
insurance_providers = {
    'ICICI Lombard General Insurance': (10000, 20000),
    'HDFC ERGO General Insurance': (15000, 25000),
    'Bajaj Allianz General Insurance': (18000, 30000),
    'Reliance General Insurance': (12000, 22000),
    'Star Health and Allied Insurance': (9000, 18000),
    'SBI General Insurance': (16000, 28000),
    'United India Insurance Company': (11000, 20000),
    'Oriental Insurance Company': (13000, 23000),
    'New India Assurance Company': (14000, 25000),
    'National Insurance Company': (17000, 27000)
}

# Page configuration
st.set_page_config(page_title='Health Insurance Premium Prediction', page_icon=':hospital:', layout='wide')

# Sidebar
st.sidebar.image(r"C:\Capstone\insurence.png", use_container_width=True)
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ('Home', 'Predict', 'Train Model', 'BMI Calculator', 'Recommendation', 'About'))

# Home Page
if page == 'Home':
    st.markdown("<h1 style='color: #6495ED;'>Welcome to Health Insurance Premium Prediction App</h1>", unsafe_allow_html=True)
    st.write("This app allows you to predict health insurance premiums based on various factors such as age, BMI, smoker status, and region.")
    st.write("Key Features:")
    st.write("- Predict insurance premiums using machine learning models.")
    st.write("- Train the model with your dataset.")
    st.write("- Calculate your BMI.")
    st.write("- Get personalized insurance provider recommendations.")

# Train Model Page
elif page == 'Train Model':
    st.title("Train Model")
    st.write("Click the button below to train the model. Ensure the required dataset and configuration are set up correctly.")
    
    if st.button("Train Model"):
        try:
            os.system("python main.py")  # Adjust to the script handling your training
            st.success("Model training completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")

# Prediction Page
elif page == 'Predict':
    st.title("Insurance Expense Prediction")
    st.header("Enter Insurance Details:")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        sex = st.selectbox("Sex", ["male", "female"])
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        bmi = calculate_bmi(weight, height)
        st.write(f"Calculated BMI: {bmi:.2f}")
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            sex = 1 if sex == "female" else 0
            smoker = 1 if smoker == "yes" else 0
            region_northeast = 1 if region == "northeast" else 0
            region_southeast = 1 if region == "southeast" else 0
            region_southwest = 1 if region == "southwest" else 0
            region_northwest = 1 if region == "northwest" else 0

            input_data = {
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region_northwest': [region_northwest],
                'region_southeast': [region_southeast],
                'region_southwest': [region_southwest]
            }

            input_df = pd.DataFrame(input_data)
            obj = PredictionPipeline()
            prediction = obj.predict(input_df)

            if prediction is None:
                st.error("Prediction failed. Check your model and inputs.")
            else:
                st.success(f"Predicted Insurance Expenses: ₹{prediction[0]:.2f}")
                st.session_state['predicted_cost'] = prediction[0]

        except Exception as e:
            st.error(f"An error occurred: {e}")

# BMI Calculation Page
elif page == 'BMI Calculator':
    st.title("BMI Calculation")
    weight = st.number_input("Enter weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Enter height (cm)", min_value=100.0, max_value=250.0, value=170.0)

    if st.button("Calculate BMI"):
        bmi = calculate_bmi(weight, height)
        st.write(f"Your BMI is: {bmi:.2f}")

# Recommendation Page
elif page == 'Recommendation':
    st.title("Insurance Provider Recommendations")
    predicted_cost = st.session_state.get('predicted_cost')

    if predicted_cost:
        st.write(f"Based on your predicted insurance cost of ₹{predicted_cost:.2f}, here are some recommended providers:")
        filtered_providers = {provider: cost_range for provider, cost_range in insurance_providers.items() if cost_range[0] <= predicted_cost <= cost_range[1]}

        if filtered_providers:
            for provider, cost_range in filtered_providers.items():
                st.write(f"{provider}: ₹{cost_range[0]} to ₹{cost_range[1]}")
        else:
            # Fallback: Recommend providers with the closest ranges
            closest_providers = sorted(insurance_providers.items(), key=lambda x: abs(x[1][0] - predicted_cost))
            st.write("No providers found within the predicted range. Here are some closest matches:")
            for provider, cost_range in closest_providers[:3]:  # Show top 3 closest matches
                st.write(f"{provider}: ₹{cost_range[0]} to ₹{cost_range[1]}")
    else:
        st.write("Please make a prediction first on the 'Predict' page.")


# About Page
elif page == 'About':
    st.title("About Health Insurance Premium Prediction App")
    st.write("This app predicts insurance premiums and provides BMI calculations and insurance provider recommendations.")
