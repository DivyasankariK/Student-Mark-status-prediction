import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open("Student_status_prediction.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'Student_status_prediction.pkl' not found. Please upload it to your repository.")

st.title("Student Semester Status Prediction")
st.write("Enter the student details below to predict their semester status.")

# Input fields
assignments = st.number_input("Assignments Score", min_value=0, max_value=100, value=70)
performance = st.number_input("Performance Score", min_value=0, max_value=100, value=70)
attendance = st.number_input("Attendance Score", min_value=0, max_value=100, value=75)
internal_status = st.selectbox("Internal Status (0 for Bad, 1 for Good)", options=[0, 1])

# Predict button
if st.button("Predict Status"):
    # Features must match the exact training column order: 
    # ['assignments', 'Performance', 'attendence', 'Internal status']
    features = np.array([[assignments, performance, attendance, internal_status]])
    
    prediction = model.predict(features)
    
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("Good State")
    else:
        st.success("Bad State")

