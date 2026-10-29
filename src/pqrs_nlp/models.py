"""Pipelines sklearn para referencia, baseline y modelo propuesto."""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from .preprocessing import TextNormalizer


def build_models(seed=42):
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, lowercase=False
    )
    word_and_char = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, lowercase=False
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
            min_df=1, max_features=40000,
        )),
    ])
    return {
        "dummy": Pipeline([
            ("normalize", TextNormalizer()),
            ("tfidf", TfidfVectorizer(lowercase=False)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("normalize", TextNormalizer()),
            ("tfidf", word_vectorizer),
            ("classifier", LogisticRegression(
                max_iter=2000, random_state=seed
            )),
        ]),
        "word_char_linear_svc": Pipeline([
            ("normalize", TextNormalizer()),
            ("features", word_and_char),
            ("classifier", LinearSVC(random_state=seed)),
        ]),
    }


def parameter_grids():
    return {
        "logistic_regression": {
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"],
        },
        "word_char_linear_svc": {
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"],
        },
    }
