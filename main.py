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

_clf = None
_feature_order: List[str] = []


def _load_model():
    global _clf, _feature_order
    if _clf is None:
        if not MODEL_PATH.exists() or not FEATURE_ORDER_PATH.exists():
            raise RuntimeError(
                f"model.pkl / feature_order.pkl not found in {BASE_DIR}. "
                "Run train.py first."
            )
        _clf = joblib.load(MODEL_PATH)
        _feature_order = joblib.load(FEATURE_ORDER_PATH)
    return _clf, _feature_order


class AnalyzeRequest(BaseModel):
    code: str


class AnalyzeResponse(BaseModel):
    algorithm: str
    confidence: float
    complexity: Dict[str, str]
    features: Dict[str, float]


@app.on_event("startup")
def load_model_on_startup():
    # Fail fast and loudly if the model isn't trained yet, rather than
    # erroring on the first request.
    _load_model()


@app.get("/")
def root():
    return {"status": "ok", "service": "AlgoLens Analysis API"}


@app.get("/health")
def health():
    try:
        _load_model()
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

    clf, feature_order = _load_model()

    row = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
    prediction = clf.predict(row)[0]
    confidence = float(clf.predict_proba(row).max())

    complexity = estimate_complexity_from_features(features)

    return AnalyzeResponse(
        algorithm=prediction,
        confidence=round(confidence, 4),
        complexity=complexity,
        features=features,
    )