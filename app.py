import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.title("Heart Disease Prediction System")
st.write("Enter the patient details and click Predict.")

model = pickle.load(open("heart_trained_model.pkl", "rb"))

chest_pain_types = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
sex_label_encoder = LabelEncoder().fit(["Male", "Female"])
chestpain_label_encoder = LabelEncoder().fit(chest_pain_types)

age = st.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", ["Male", "Female"])
chestpain = st.selectbox("Chest Pain Type", chest_pain_types)
maxheartrate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=130)

if st.button("Predict"):
    features = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_label_encoder.transform([sex])[0]],
        "ChestPain": [chestpain_label_encoder.transform([chestpain])[0]],
        "MaxHeartRate": [maxheartrate],
    })
    result = model.predict(features)[0]
    if result == 1:
        st.error("Prediction: HEART DISEASE")
    else:
        st.success("Prediction: NO HEART DISEASE")
