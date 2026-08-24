"""
main.py - FastAPI service for AlgoLens.

Exposes a single POST /analyze endpoint that takes Python source code and
returns:
  - the predicted algorithm category (from the trained Random Forest)
  - a confidence score
  - a rule-based time/space complexity estimate with reasoning
  - the raw extracted feature vector (useful for the frontend's
    "Algorithm Identified" tab explanation and for debugging)

Run locally:
    uvicorn main:app --reload
Then test with:
    curl -X POST http://127.0.0.1:8000/analyze \
         -H "Content-Type: application/json" \
         -d "{\"code\": \"def f(x): return x\"}"
"""

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_extractor import extract_features
from complexity_estimator import estimate_complexity_from_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.pkl"

app = FastAPI(title="AlgoLens Analysis API")

# Allow the Lovable frontend (and local dev) to call this API directly.
# Tighten allow_origins to your actual deployed frontend URL before
# shipping - "*" is fine for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from model import load_model, predict_algorithm


class AnalyzeRequest(BaseModel):
    code: str


class AnalyzeResponse(BaseModel):
    algorithm: str
    algorithm_name: str
    detected_pattern: str
    confidence: float
    complexity: Dict[str, str]
    features: Dict[str, float]


@app.on_event("startup")
def load_model_on_startup():
    load_model()


@app.get("/")
def root():
    return {"status": "ok", "service": "AlgoLens Analysis API"}


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "model_loaded": False, "detail": str(e)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    code = payload.code
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="`code` field is empty.")

    try:
        features = extract_features(code)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Could not parse code: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {e}")

    prediction, confidence = predict_algorithm(code, features)
    complexity = estimate_complexity_from_features(features)

    return AnalyzeResponse(
        algorithm=prediction,
        algorithm_name=prediction,
        detected_pattern=f"AlgoLens ML — {prediction}",
        confidence=round(confidence, 4),
        complexity=complexity,
        features=features,
    )