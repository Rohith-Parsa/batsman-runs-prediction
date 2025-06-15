import streamlit as st
import numpy as np
import joblib

# Load model and scaler
rf = joblib.load("runs_model.pkl")        
scaler = joblib.load("runs_scaler.pkl")   
st.title(" Predict Total Career Runs")
st.markdown("Enter the stats below:")

bf = st.number_input("Balls Faced (BF)", min_value=0)
fours = st.number_input("Number of 4s", min_value=0)
sixes = st.number_input("Number of 6s", min_value=0)
innings = st.number_input("Innings Played", min_value=0)
strike_rate = st.number_input("Strike rate", min_value=0)

if st.button("Predict Runs"):
    input_data = np.array([[bf, fours, sixes, innings, strike_rate]])

    input_scaled = scaler.transform(input_data)

    prediction = rf.predict(input_scaled)[0]

    st.success(f" Predicted Total Runs: {round(prediction)}")
