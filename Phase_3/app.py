import streamlit as st
import joblib
import numpy as np


score_model = joblib.load("score_model.pkl")
result_model = joblib.load("result_model.pkl")

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.markdown("### 📋 Enter Student Details")


col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("📚 Study Hours", min_value=1.0, max_value=24.0)
    attendance = st.number_input("📊 Attendance (%)", min_value=1.0, max_value=100.0)
    internal_score = st.number_input("📝 Internal Score (out of 50)", min_value=1.0, max_value=50.0)

with col2:
    previous_result = st.number_input("📈 Previous Result (out of 100)", min_value=1.0, max_value=100.0)
    internet_usage = st.number_input("🌐 Internet Usage (hrs/day)", min_value=1.0, max_value=24.0)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Performance"):

    try:
        input_data = np.array([[ 
            study_hours,
            attendance,
            internal_score,
            previous_result,
            internet_usage
        ]])

      
        predicted_score = score_model.predict(input_data)[0]
        predicted_class = result_model.predict(input_data)[0]

        # Convert classifier output
        if predicted_class in [1, "Pass", "PASS"]:
            result_text = "PASS"
        else:
            result_text = "FAIL"

        predicted_score = max(0, min(100, predicted_score))

        # ---------------- OUTPUT ----------------
        st.markdown("## 📊 Results")
        st.progress(predicted_score / 100)

        st.success(f"🎯 Predicted Score: {round(predicted_score, 2)} / 100")
        st.success(f"🧠 Model Prediction: {result_text}")

    except Exception as e:
        st.error(f"❌ Error: {e}")