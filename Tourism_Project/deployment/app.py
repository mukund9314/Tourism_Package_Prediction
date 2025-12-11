import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


# Load trained model from Hugging Face
model_path = hf_hub_download(
    repo_id="mukund9314/Tourism-Package-Model",
    filename="best_tourism_package_prediction_model_v1.joblib",
    repo_type="model"
)

model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase a tourism package
based on demographic, income, and travel-related characteristics.
Please enter the customer information below.
""")


# Feature Lists
numeric_features = [
    'Age','DurationOfPitch','NumberOfPersonVisiting',
    'NumberOfFollowups','NumberOfTrips','PitchSatisfactionScore',
    'NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    "TypeofContact","Occupation","Gender","ProductPitched",
    "MaritalStatus","Designation","CityTier",
    "PreferredPropertyStar","Passport","OwnCar"
]


# Dropdown options based on dataset

options_dict = {
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Gender": ["Male", "Female"],
    "ProductPitched": ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    "CityTier": [1, 2, 3],
    "PreferredPropertyStar": [3, 4, 5],
    "Passport": [0, 1],
    "OwnCar": [0, 1]
}

st.header("🧑‍💼 Customer Information")

# Numeric Inputs (with realistic defaults)
numeric_inputs = {
    "Age": st.number_input("Age", min_value=18, max_value=80, value=37),
    "DurationOfPitch": st.number_input("Duration Of Pitch (minutes)", min_value=1, max_value=150, value=15),
    "NumberOfPersonVisiting": st.number_input("Number of Persons Visiting", min_value=1, max_value=6, value=3),
    "NumberOfFollowups": st.number_input("Number Of Follow-ups", min_value=0, max_value=20, value=3),
    "NumberOfTrips": st.number_input("Number Of Trips Per Year", min_value=0, max_value=30, value=3),
    "PitchSatisfactionScore": st.number_input("Pitch Satisfaction Score (1–5)", min_value=1, max_value=5, value=3),
    "NumberOfChildrenVisiting": st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=1),
    "MonthlyIncome": st.number_input("Monthly Income (INR)", min_value=1000, max_value=200000, value=23000)
}

# Categorical Inputs
categorical_inputs = {}
for feature in categorical_features:
    categorical_inputs[feature] = st.selectbox(
        label=feature,
        options=options_dict[feature]
    )

# Combine all inputs into a DataFrame
input_data = {**numeric_inputs, **categorical_inputs}
input_df = pd.DataFrame([input_data])


# Predict Button
if st.button("Predict Purchase"):
    prediction = model.predict(input_df)[0]
    result = (
        "Customer is **LIKELY** to purchase the tourism package."
        if prediction == 1
        else "Customer is **NOT** likely to purchase the tourism package."
    )

    st.subheader("🔍 Prediction Result")
    st.success(result)
