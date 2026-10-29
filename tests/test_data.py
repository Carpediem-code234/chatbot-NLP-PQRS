import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.constants import INTENTS, REQUIRED_COLUMNS
from src.data_validation import load_dataset, normalize_for_dedup, validate

DATA = ROOT / "data/raw/pqrs_sintetico.csv"


@pytest.fixture(scope="module")
def df():
    return load_dataset(str(DATA))


def test_columns_and_size(df):
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 658


def test_labels_and_splits(df):
    assert sorted(df["intent"].unique()) == sorted(INTENTS)
    assert df["split"].value_counts().to_dict() == {"train": 398, "dev": 127, "test": 133}


def test_every_intent_in_every_split(df):
    ct = pd.crosstab(df["intent"], df["split"])
    assert (ct.values > 0).all()


def test_no_template_leakage(df):
    assert (df.groupby("template_id")["split"].nunique() == 1).all()


def test_no_exact_cross_split_duplicates(df):
    n = df["text"].map(normalize_for_dedup)
    assert (df.assign(n=n).groupby("n")["split"].nunique() == 1).all()


def test_no_label_conflict_same_text(df):
    n = df["text"].map(normalize_for_dedup)
    assert (df.assign(n=n).groupby("n")["intent"].nunique() == 1).all()


def test_jsonl_matches_csv(df):
    j = load_dataset(str(ROOT / "data/raw/pqrs_sintetico.jsonl"))
    assert j[REQUIRED_COLUMNS].astype(str).equals(df[REQUIRED_COLUMNS].astype(str))


def test_validate_reports_no_critical(df):
    assert validate(df)["resumen"]["critical"] == 0


def test_validate_detects_template_leak(df):
    bad = df.copy()
    i = bad.index[bad["split"] == "test"][0]
    bad.loc[i, "template_id"] = bad.loc[bad["split"] == "train", "template_id"].iloc[0]
    assert validate(bad)["checks"]["fuga_template_id"]["status"] == "critical"


def test_validate_detects_invalid_label_and_empty(df):
    bad = df.copy()
    bad.loc[0, "intent"] = "inventada"
    bad.loc[1, "text"] = "   "
    r = validate(bad)["checks"]
    assert r["etiquetas_validas"]["status"] == "critical"
    assert r["textos_vacios"]["status"] == "critical"


def test_validate_detects_missing_column(df):
    r = validate(df.drop(columns=["split"]))
    assert r["checks"]["columnas"]["status"] == "critical"


def test_dedup_normalization_keeps_text_untouched(df):
    before = df["text"].copy()
    df["text"].map(normalize_for_dedup)
    assert before.equals(df["text"])
    assert normalize_for_dedup("¡NO me atendieron!") == "no me atendieron"
