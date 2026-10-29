"""Preprocesamiento mínimo y conservador.

Decisión (spec §5): NO se eliminan negaciones, tildes ni signos que puedan
cambiar la intención. Solo se normaliza Unicode (NFC), se recortan espacios y
se colapsan espacios repetidos. La minúscula la aplica el vectorizador.
No se usa lista de stopwords (eliminaría «no», «sin», «nada», «ya»...).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def select_train_dev(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo train y dev: el test no debe existir en el proceso de entrenamiento/selección."""
    return df[df["split"].isin(["train", "dev"])].copy()


def get_xy(df: pd.DataFrame, split: str):
    part = df[df["split"] == split]
    return part["text"].map(clean_text).tolist(), part["intent"].tolist(), part
