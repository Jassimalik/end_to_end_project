import streamlit as st
import sys
import os

# Add src directory to system path
sys.path.append(os.path.abspath("src"))

# Import your prediction classes
from mlproject.pipelines.prediction_pipeline import DataPredition,CustomData

# Streamlit app title
st.title("Student Math Score Predictor")

# Input fields
gender = st.selectbox("Select gender:", ["male", "female"])
race_ethnicity = st.selectbox('Select race ethnicity', ["group A", "group B", "group C", "group D"])
parental_level_of_education = st.selectbox("Select parental level of education", [
    "bachelor's degree", 'some college', "master's degree",
    "associate's degree", 'high school', 'some high school'
])
lunch = st.selectbox('Select lunch', ['standard', 'free/reduced'])
test_preparation_course = st.selectbox('Select test preparation course', ['none', 'completed'])
reading_score = st.number_input('Enter reading score:', min_value=0, max_value=100)
writing_score = st.number_input('Enter writing score:', min_value=0, max_value=100)

# Predict button
if st.button("Predict math score"):
    # Create custom data object
    custom_data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert input data to DataFrame
    input_df = custom_data.data_as_dataframe()

    # Run prediction
    prediction = DataPredition().predition(input_df)

    # Show prediction
    st.success(f"Predicted math score: {prediction}")
