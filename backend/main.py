from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import Optional
import os

# ── charger le pipeline entraîné ─────────────────────────────
MODEL_PATH = "./airbnb_demand_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # modèle fictif pour les tests
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.classes_ = [0, 1]
    def dummy_predict_proba(X):
        return np.array([[0.3, 0.7]] * len(X))  # fixe une proba > 0.5
    model.predict_proba = dummy_predict_proba

# - si le pipeline garde la trace des noms de colonnes (v1.2+)
if hasattr(model, "feature_names_in_"):
    EXPECTED_COLS = list(model.feature_names_in_)
else:
    EXPECTED_COLS = [
        "neighbourhood_group", "neighbourhood", "room_type",
        "minimum_nights", "number_of_reviews", "availability_365",
        "latitude", "longitude", "reviews_per_month",
        "last_review_year", "last_review_month", "last_review_dayofweek",
        "calculated_host_listings_count", "never_reviewed",
        "id", "host_id", "name", "host_name", "price"
    ]

# ── FastAPI ─────────────────────────────────────────────────
app = FastAPI(title="Airbnb Demand API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── schéma d’entrée ─────────────────────────────────────────
class ListingFeatures(BaseModel):
    neighbourhood_group: str          = Field(examples=["Manhattan"])
    room_type: str                    = Field(examples=["Entire home/apt"])
    minimum_nights: int               = Field(examples=[3])
    number_of_reviews: int            = Field(examples=[12])
    availability_365: int             = Field(examples=[200])
    latitude: float                   = Field(examples=[40.735])
    longitude: float                  = Field(examples=[-73.99])
    reviews_per_month: float          = Field(examples=[0.8])
    last_review_year: int             = Field(examples=[2023])
    last_review_month: int            = Field(examples=[5])
    last_review_dayofweek: int        = Field(examples=[4])

# ── helpers ─────────────────────────────────────────────────
def neutral_value(colname: str):
    if colname in ["neighbourhood_group", "neighbourhood", "room_type", "name", "host_name"]:
        return "Unknown"
    return 0

# ── endpoint prédiction ─────────────────────────────────────
@app.post("/predict")
def predict(features: ListingFeatures):
    df = pd.DataFrame([features.dict()])
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = neutral_value(col)
    df = df[EXPECTED_COLS]
    proba = model.predict_proba(df)[0, 1]
    return {
        "high_demand": int(proba >= 0.5),
        "probability": round(float(proba), 3)
    }

@app.get("/")
def root():
    return {"status": "API OK"}
