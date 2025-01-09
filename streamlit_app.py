import streamlit as st
import requests

# API Endpoint
LIGHTGBM_API = "http://localhost:8080/predict/lightgbm/"
CODEBERT_API = "http://localhost:8080/predict/codebert/"

st.title("AI Detected Score Prediction")

# Input fields
question_code = st.text_input("Question Code")
candidate_answer = st.text_area("Candidate Answer")
ai_generated_answer = st.text_area("AI Generated Answer")

if st.button("Predict with LightGBM"):
    if question_code and candidate_answer and ai_generated_answer:
        payload = {
            "question_code": question_code,
            "candidate_answer": candidate_answer,
            "ai_generated_answer": ai_generated_answer
        }
        try:
            response = requests.post(LIGHTGBM_API, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Score (LightGBM): {result['predicted_score']:.4f}")
                st.info(f"Interpretation: {result['interpretation']}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please fill in all input fields.")

if st.button("Predict with CodeBERT"):
    response = requests.post(
        CODEBERT_API,
        json={
            "question_code": question_code,
            "candidate_answer": candidate_answer,
            "ai_generated_answer": ai_generated_answer
        },
    )
    if response.status_code == 200:
        st.success(f"Predicted Score (CodeBERT): {response.json()['predicted_score']}")
    else:
        st.error(f"Error: {response.json()['detail']}")
