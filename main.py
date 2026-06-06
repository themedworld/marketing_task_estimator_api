from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import os

# =====================================
# CONFIG
# =====================================

MODEL_PATH = "./best_duration_model.joblib"
MLB_PATH   = "./mlb_channels.joblib"        # ✅ ajouté

# =====================================
# APP
# =====================================

app = FastAPI(title="Marketing Duration Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# LOAD MODEL + MLB
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(MLB_PATH):
    raise FileNotFoundError(f"MLB not found: {MLB_PATH}")

model = joblib.load(MODEL_PATH)
mlb   = joblib.load(MLB_PATH)               # ✅ ajouté

print("Model loaded successfully")

# =====================================
# INPUT SCHEMA
# =====================================

class PredictInput(BaseModel):
    data: Dict[str, Any]

# =====================================
# ROUTES
# =====================================

@app.get("/")
def home():
    return {"message": "API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================
# PREDICT
# =====================================

@app.post("/predict")
def predict(input_data: PredictInput):
    try:
        single_example = input_data.data

        print(
            f"[predict] task_type={single_example.get('task_type')} "
            f"status={single_example.get('status')} "
            f"priority={single_example.get('priority')} "
            f"channel_group={single_example.get('channel_group')}"
        )

        # =========================
        # Expected columns
        # =========================

        expected_cols = [
            'task_type', 'status', 'priority',
            'cost', 'impressions', 'clicks', 'conversions', 'leads',
            'score', 'ctr', 'channel_group', 'complexity', 'effort',
            'unused1', 'flag', 'id',
            'ch_App', 'ch_Bing', 'ch_Creative', 'ch_Display', 'ch_Email',
            'ch_Facebook', 'ch_Google', 'ch_Instagram', 'ch_Internal',
            'ch_LinkedIn', 'ch_Multi', 'ch_Native', 'ch_Pinterest',
            'ch_Programmatic', 'ch_Push', 'ch_SMS', 'ch_Snapchat',
            'ch_Social', 'ch_Spotify', 'ch_TV', 'ch_TikTok', 'ch_Twitter',
            'ch_Website', 'ch_YouTube',
        ]

        # =========================
        # DataFrame
        # =========================

        raw = pd.DataFrame([single_example]).copy()

        # Ajouter colonnes manquantes
        for c in expected_cols:
            if c not in raw.columns:
                raw[c] = np.nan

        # =========================
        # ✅ Handle channels via mlb (même logique que l'entraînement)
        # =========================

        channels_str = str(single_example.get("channels", ""))
        channels_list = [[s.strip() for s in channels_str.split(";") if s.strip()]]

        ch_df = pd.DataFrame(
            mlb.transform(channels_list),
            columns=[f"ch_{c}" for c in mlb.classes_],
        )

        for col in ch_df.columns:
            raw[col] = ch_df[col].values[0]

        # Supprimer la colonne channels brute
        if "channels" in raw.columns:
            raw = raw.drop(columns=["channels"])

        # Réordonner
        raw = raw[expected_cols]

        # =========================
        # Prediction
        # =========================

        prediction = model.predict(raw)

        if hasattr(prediction, "ndim") and prediction.ndim > 1:
            prediction = prediction.ravel()

        result = float(prediction[0])
        print(f"[predict] résultat: {result:.2f}h")

        return {"prediction": result}

    except Exception as e:
        print(f"[predict] erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))