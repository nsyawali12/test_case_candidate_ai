from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.utils.model import LightGBMmodel
# from app.utils.fine_tune_codebert import CodeBERTFineTuner
from app.utils.similarities import compute_similarity_metrics
import pickle

router = APIRouter()

#loading lgbm model
try:
    with open("app/models/lightgbm_model_100r.pkl", "rb") as file:
        lgbm_model = pickle.load(file)
except FileNotFoundError as e:
    raise RuntimeError("LightGBM model file not found!") from e

# loading codebert model (wait until had cloud )
# codebert_model = CodeBERTFineTuner(model_path="app/models/codebert_finetuned_60e")


class PredictInput(BaseModel):
    question_code: str
    candidate_answer: str
    ai_generated_answer: str

@router.post("/predict/lightgbm/")
def predict_lightgbm(input_data: PredictInput):
    try:
        # Compute similarity metrics using the imported function
        similarities = compute_similarity_metrics(
            input_data.candidate_answer, input_data.ai_generated_answer
        )
        
        # Prepare features DataFrame
        features_df = pd.DataFrame([similarities])
        
        # Ensure the correct column order matches training
        required_columns = ["cosine_similarity", "jaccard_similarity", "levenshtein_similarity"]
        if not all(col in features_df.columns for col in required_columns):
            raise ValueError(f"Features DataFrame must include columns: {required_columns}")
        
        # Predict using the LightGBM model
        predicted_score = lgbm_model.predict(features_df)[0]

        # Add interpretation
        interpretation = "More likely human" if predicted_score < 0.5 else "More likely AI"
        
        # Return response
        return {
            "question_code": input_data.question_code,
            "predicted_score": predicted_score,
            "interpretation": interpretation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/predict/codebert/")
# def predict_codebert(input_data: PredictInput):
#     try:
#         predicted_score = codebert_model.predict(
#             input_data.candidate_answer, input_data.ai_generated_answer
#         )
#         return {
#             "question_code": input_data.question_code,
#             "predicted_score": predicted_score,
#             "interpretation": "More likely AI" if predicted_score >= 0.5 else "More likely human",
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))