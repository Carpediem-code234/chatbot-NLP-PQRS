"""Definición de los modelos comparados en la Fase 2.

M0  Clase mayoritaria (piso de referencia)
M1  Baseline: TF-IDF de palabras sobre texto crudo + regresión logística
M2  TF-IDF de palabras + Naive Bayes complementario
M3  Propuesto: texto normalizado → TF-IDF palabras ∪ n-gramas de caracteres ∪ rasgos estructurales
    → SVM lineal calibrada (probabilidades para el umbral de la regla R2)
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import HP_M1, HP_M3, SEED
from src.preprocesamiento import COLS_META


class Columna(BaseEstimator, TransformerMixin):
    """Selecciona una columna de texto o un bloque de columnas numéricas de un DataFrame."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]


def m0_mayoritaria():
    return Pipeline([("sel", Columna("texto")), ("clf", DummyClassifier(strategy="most_frequent"))])


def m1_baseline(C=HP_M1["C"], ngram_range=HP_M1["ngram_range"]):
    return Pipeline([("sel", Columna("texto")),
                     ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram_range, sublinear_tf=True)),
                     ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED))])


def m2_nb(alpha=0.05):
    return Pipeline([("sel", Columna("texto")),
                     ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
                     ("clf", ComplementNB(alpha=alpha))])


def m3_propuesto(C=HP_M3["C"], char_rng=HP_M3["char_rng"], usar_meta=HP_M3["usar_meta"], texto="texto_norm"):
    ramas = [("palabras", Pipeline([("sel", Columna(texto)),
                                    ("tf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1))])),
             ("caracteres", Pipeline([("sel", Columna(texto)),
                                      ("tf", TfidfVectorizer(analyzer="char_wb", ngram_range=char_rng,
                                                             sublinear_tf=True, min_df=2))]))]
    if usar_meta:
        ramas.append(("meta", Pipeline([("sel", Columna(COLS_META)), ("esc", StandardScaler())])))
    svm = LinearSVC(C=C, class_weight="balanced", max_iter=20000, random_state=SEED)
    return Pipeline([("feats", FeatureUnion(ramas)),
                     ("clf", CalibratedClassifierCV(svm, method="sigmoid", cv=3))])


MODELOS = {"M0 Mayoritaria": m0_mayoritaria, "M1 Baseline": m1_baseline,
           "M2 ComplementNB": m2_nb, "M3 Propuesto": m3_propuesto}
