import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Diabetes prediction",
    layout="centered",
    initial_sidebar_state="auto"
)



MODEL_FILENAME = 'diabetes_model.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"❌ Error: Model file '{MODEL_FILENAME}' not found.")
        st.info("Please ensure you have saved the model in a Jupyter Notebook using joblib.dump(lrm, 'diabetes_model.pkl') and that the file is in the same folder.")
        return None
    
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model (variable name lrn is preserved as per request)
lrn = load_model()


st.title("🩺 Diabetes prediction")

st.write("Enter the following parameters to assess the risk of developing diabetes:")

if lrn is not None:
    
    FEATURE_NAMES = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            '1. Number of pregnancies',
            min_value=0, max_value=20, value=1, step=1,
            help="Number of pregnancies"
        )
        glucose = st.number_input(
            '2. Glucose level (mg/dL)',
            min_value=0.0, max_value=200.0, value=120.0, step=0.1,
            help="Plasma glucose concentration"
        )
        blood_pressure = st.number_input(
            '3. Blood pressure (mm Hg)',
            min_value=0.0, max_value=150.0, value=70.0, step=0.1,
            help="Diastolic blood pressure"
        )
        skin_thickness = st.number_input(
            '4. Skin thickness (mm)',
            min_value=0.0, max_value=100.0, value=25.0, step=0.1,
            help="Triceps skin fold thickness"
        )
        
    with col2:
        insulin = st.number_input(
            '5. Insulin Level (µU/ml)',
            min_value=0.0, max_value=900.0, value=0.0, step=0.1,
            help="2-hour serum insulin"
        )
        bmi = st.number_input(
            '6. Body mass index (BMI) (kg/m²)',
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Body Mass Index"
        )
        dpf = st.number_input(
            '7. Diabetes pedigree function',
            min_value=0.0, max_value=2.5, value=0.4, step=0.001, format="%.3f",
            help="Genetic risk assessment"
        )
        age = st.number_input(
            '8. Age',
            min_value=0, max_value=120, value=30, step=1,
            help="Age of the person in years"
        )
        
    st.markdown(" ")
    
    predict_button = st.button("🔍 Get prediction", type="primary", use_container_width=True)

    if predict_button:
        input_data = [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]
        
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # Perform prediction
        prediction = lrn.predict(input_df)[0]
        
        st.header("Analysis result:")
        
        if prediction == 1:
            st.error("🔴 **POSITIVE RESULT**")
            st.markdown("The model predicts a **high risk** of diabetes. Medical consultation is recommended")
        else:
            st.success("🟢 **NEGATIVE RESULT**")
            st.markdown("The model predicts a **low risk** of diabetes. Continue to monitor your health")
            
        # Get probability for class 1 (diabetes)
        probability = lrn.predict_proba(input_df)[0]
        st.caption(f"Probability of Diabetes (Class 1): **{probability[1]*100:.2f}%**")
        
    st.markdown("---")


st.sidebar.title("ATTENTION!")
st.sidebar.info("This site is a demonstration of a machine learning model and is not a medical tool. The results obtained do not replace professional diagnosis or consultation with a specialist!")
