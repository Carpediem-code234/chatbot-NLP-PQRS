"""Normalización conservadora de mensajes en español."""

import re

import pandas as pd


def normalize_text(value):
    """Normaliza espacios y mayúsculas sin borrar tildes, negaciones ni signos."""
    if pd.isna(value):
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip().casefold()


class TextNormalizer:
    """Transformador sklearn compatible para normalizar cada texto."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_text(value) for value in X]
