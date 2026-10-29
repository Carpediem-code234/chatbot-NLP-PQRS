import pandas as pd

from pqrs_nlp.models import build_models
from pqrs_nlp.preprocessing import normalize_text


def test_normalization_preserves_negation_and_accents():
    assert normalize_text("  NO   recibí  el medicamento\t") == "no recibí el medicamento"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text(pd.NA) == ""


def test_all_pipelines_fit_and_predict_text():
    X = pd.Series([
        "Necesito una cita con cardiología",
        "Quiero quejarme por la atención recibida",
        "No me entregan el medicamento",
        "Quiero consultar mi autorización",
        "Solicito una cita de medicina general",
        "Me atendieron muy mal y quiero una queja",
        "Solicito entrega de insulina",
        "¿Ya está lista mi autorización?",
    ])
    y = pd.Series([
        "solicitar_cita", "queja_atencion", "reclamo_medicamentos",
        "consultar_autorizacion", "solicitar_cita", "queja_atencion",
        "reclamo_medicamentos", "consultar_autorizacion",
    ])
    for model in build_models().values():
        model.fit(X, y)
        predictions = model.predict(X[:2])
        assert len(predictions) == 2
        assert set(predictions) <= set(y)
