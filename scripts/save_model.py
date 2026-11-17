import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline_model import build_pipeline

df = pd.read_csv(ROOT / "data/synthetic/raw/pqrs_sintetico.csv")
train = df[df["split"].isin(["train", "dev"])]  # NO uses test

# Usa los hiperparámetros ganadores de tu tuning; estos son de ejemplo
model = build_pipeline(features="word+char", ngram="large", C=10.0, class_weight="balanced")
model.fit(train["text"], train["intent"])

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models/baseline.joblib")
print("Modelo guardado")