import pickle
import numpy as np
import streamlit as st
from main import model


def load_model():
    with (open("main.py")) as f:
        data = pickle.load(f.encode())
    return data


# Website
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.sidebar.markdown("""
* **age** - The age of the patient.
* **sex** - The gender of the patient. (1 = male, 0 = female).
* **cp** - Type of chest pain. (0 = typical angina, 1 = atypical angina, 2 = non — anginal pain, 3 = asymptotic).
* **trestbps** - Resting blood pressure in mmHg.
* **chol** - Serum Cholesterol in mg/dl.
* **fbs** - Fasting Blood Sugar. (1 = fasting blood sugar is more than 120mg/dl, 0 = otherwise).
* **restecg** - Resting ElectroCardioGraphic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hyperthrophy).
* **thalach** - Max heart rate achieved.
* **exang** - Exercise induced angina (1 = yes, 0 = no).
* **oldpeak** - ST depression induced by exercise relative to rest.
* **slope** - Peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).
* **ca** - Number of major vessels (0–3) colored by fluoroscopy.
* **thal** - Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect).
* **target** - Diagnosis of heart disease (0 = absence, 1 = present)""")

# Main Page
st.title("Heart Disease Predictor")
st.subheader("This model attempts to predict the possibility of having heart disease using the Logistic Regression"
             "algorithm. Collected data is standardized.")
st.subheader("The model was trained on the University of California Irvine's heart disease dataset from Kaggle and "
             "has 85.1% accuracy")
st.subheader("Please note however that this is for educative purposes only and should NOT be used as a medical "
             "diagnosis. Please consult your local medical professional if you have any health concerns.")

# Collecting user data
st.write("Please enter information below")

age = st.slider("Age")

sex_values = {"Male": 1, "Female": 0}
sex = st.selectbox("Sex", sex_values)

cp_values = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = st.selectbox("Chest Pain Type (select 'Asymptomatic' if no pain)", cp_values)

trestbps = st.slider("Resting Blood Pressure (mmHG)", max_value=220)

chol = st.slider("Serum Cholesterol (mg/dl)", max_value=500)

fbs_values = {"Over 120 mg/dl": 1, "Under 120 mg/dl": 0}
fbs = st.selectbox("Fasting Blood Sugar (mg/dl)", fbs_values)

restecg_values = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = st.selectbox("Resting ElectroCardioGraphic (ECG) Results", restecg_values)

thalach = st.slider("Maximum Heart Rate Achieved (bpm)", max_value=300)

exang_values = {"Yes": 1, "No": 0}
exang = st.selectbox("Do you have Exercise Induced Angina?", exang_values)

oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", min_value=0.0, max_value=7.0,
                    step=.1)

slope_values = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = st.selectbox("Slope (Peak exercise ST segment)", slope_values)

ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", max_value=3)

thal_values = {"Normal": 2, "Fixed Defect": 6, "Reversible Defect": 7}
thal = st.selectbox("Thalassemia", thal_values)

# Agreement
agreement = st.checkbox("I understand that this is not a medical diagnosis and that I should visit a healthcare "
                        "provider if I have any health concerns")

# Predict Button
predict_button = st.button("Get Results")

if agreement:
    if predict_button:
        input_data = {"age": [age], "sex": [sex_values[sex]], "cp": [cp_values[cp]], "trestbps": [trestbps],
                      "chol": [chol],
                      "fbs": [fbs_values[fbs]], "restecg": [restecg_values[restecg]], "thalach": [thalach],
                      "exang": [exang_values[exang]],
                      "oldpeak": [oldpeak], "slope": [slope_values[slope]], "ca": [ca], "thal": [thal_values[thal]]}

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)

    if prediction[0] == 1:
        st.write("The model predicts that you have heart disease.")
        st.write(f"Model Prediction: {prediction[0]} (1 = heart disease present, 0 = no heart disease)")
    else:
        st.write("The model predicts that you do not have heart disease.")
        st.write(f"Model Prediction: {prediction[0]} (1 = heart disease present, 0 = no heart disease)")
