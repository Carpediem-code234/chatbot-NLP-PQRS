"""Rutas y constantes compartidas del proyecto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_EXTERNAL = RAIZ / "data" / "external"
MODELS = RAIZ / "models"
RESULTS = RAIZ / "results"

DATASET_PQRS = DATA_RAW / "pqrs_salud_sintetico.csv"
REGISTROS_NUEVOS = DATA_RAW / "registros_nuevos.csv"   # agregados con #nuevo o cargue masivo
TEST_EXTERNO = DATA_RAW / "pqrs_test_externo.csv"
GLOSARIO = DATA_RAW / "glosario_salud_esp_way.csv"
GLOSARIO_AMPLIADO = DATA_RAW / "glosario_way_esp_ampliado.csv"   # generado por src/glosario_wayuu.py
MODELO = MODELS / "modelo_pqrs_m3.joblib"

SEED = 42
CLASES = ["Peticion", "Queja", "Reclamo", "Sugerencia", "Otro"]

# Hiperparámetros elegidos en la Fase 2 (GridSearchCV con GroupKFold por plantilla; ver results/experimentos.csv)
HP_M1 = {"C": 30, "ngram_range": (1, 1)}
HP_M3 = {"C": 3.0, "char_rng": (3, 5), "usar_meta": True}

# Umbral de confianza de la regla R2, elegido en validación (accuracy >= 90 % en lo que el modelo decide solo)
TAU = 0.80
