"""Baseline: TF-IDF (palabras y/o caracteres) + Regresión Logística (spec §7)."""
from __future__ import annotations

import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

NGRAMS = {  # variantes de n-gramas (pequeña / grande)
    "small": {"word": (1, 1), "char": (2, 4)},
    "large": {"word": (1, 2), "char": (3, 5)},
}


def build_pipeline(features="word+char", ngram="large", C=1.0,
                   class_weight=None, seed=42) -> Pipeline:
    ng = NGRAMS[ngram]
    # Sin stopwords, sin strip_accents; token_pattern conserva palabras de 1 letra.
    word = TfidfVectorizer(analyzer="word", ngram_range=ng["word"], lowercase=True,
                           strip_accents=None, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=ng["char"], lowercase=True,
                           strip_accents=None, sublinear_tf=True)
    parts = {"word": [("word", word)], "char": [("char", char)],
             "word+char": [("word", word), ("char", char)]}[features]
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000,
                             random_state=seed)
    return Pipeline([("feats", FeatureUnion(parts)), ("clf", clf)])


def param_grid():
    """48 configuraciones: 3 conjuntos de rasgos x 2 rangos n-grama x 4 C x 2 pesos de clase.
    Justificación: con ~400 ejemplos de entrenamiento una malla mayor sobreajustaría dev."""
    for f, ng, C, cw in itertools.product(
            ["word", "char", "word+char"], ["small", "large"],
            [0.3, 1.0, 10.0, 100.0], [None, "balanced"]):
        yield {"features": f, "ngram": ng, "C": C, "class_weight": cw}
