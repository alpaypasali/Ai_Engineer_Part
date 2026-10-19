from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import predict_complaint


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="ComplaintRouter AI",
    description="Financial complaint classification API",
    version="1.0.0"
)


# ==========================================
# 2. PYDANTIC MODELS
# ==========================================

class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    category: str
    probability: float
    status: str


# ==========================================
# 3. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "ComplaintRouter"
    }


# ==========================================
# 4. PREDICTION ENDPOINT
# ==========================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        result = predict_complaint(request.text)

        return result

    except (ValueError, TypeError) as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable."
        ) from e