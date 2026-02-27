import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #e74c3c, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 0.6rem;
    border-radius: 12px;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #c0392b, #922b21);
    transform: scale(1.02);
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.prediction-box {
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
}

.disease {
    background-color: #ffebee;
    color: #c62828;
}

.healthy {
    background-color: #e8f5e9;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("heart_disease_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        return model_data["model"], model_data["feature_names"], scaler
    except:
        return None, None, None

model, feature_names, scaler = load_model()

# Header
st.markdown('<div class="main-title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Risk Assessment Tool</div>', unsafe_allow_html=True)
st.markdown("---")

if model is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧑 Personal Information")
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")

        st.subheader("🫀 Heart Metrics")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                         format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                               "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                          format_func=lambda x: "Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG", [0, 1, 2],
                              format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1],
                            format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [1, 2, 3],
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔍 Predict Heart Disease Risk"):

        input_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.markdown("## 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box disease">⚠️ DISEASE RISK DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box healthy">✅ HEALTHY</div>', unsafe_allow_html=True)

        with col2:
            st.metric("Confidence Score", f"{max(proba) * 100:.2f}%")

        with col3:
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            st.metric("Risk Level", risk)

        # Enhanced Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Healthy"],
            y=[proba[0]],
            text=[f"{proba[0]*100:.1f}%"],
            textposition="auto"
        ))
        fig.add_trace(go.Bar(
            x=["Disease"],
            y=[proba[1]],
            text=[f"{proba[1]*100:.1f}%"],
            textposition="auto"
        ))

        fig.update_layout(
            title="Prediction Probability Distribution",
            yaxis_title="Probability",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.error("⚠️ High Risk Detected. Please consult a cardiologist and maintain a healthy lifestyle.")
        else:
            st.success("✅ Low Risk. Continue healthy habits and regular checkups.")

        with st.expander("📋 View Input Summary"):
            st.write(input_data)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Model Information")
        st.info("""
        **Logistic Regression Model**
        - Accuracy: ~85%
        - Dataset Size: 303 Patients
        - Features Used: 13 Clinical Attributes
        """)

        st.markdown("## 🔑 Key Risk Factors")
        st.markdown("""
        - Chest pain type  
        - Age & Gender  
        - Blood pressure  
        - Cholesterol levels  
        - Max heart rate  
        - Exercise angina  
        """)

        st.markdown("---")
        st.caption("⚠️ For educational purposes only")
        st.caption("Not a substitute for professional medical advice")

else:
    st.error("Model not found. Please generate model files first.")
