from fastapi import FastAPI
from app.utils.predict import router as predict_router  # Unified prediction logic

app = FastAPI(
    title="AI Detected Score Prediction API",
    description="API for predicting AI-detected scores using LightGBM and CodeBERT.",
    version="1.0.0"
)

# Include prediction routes
app.include_router(predict_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)