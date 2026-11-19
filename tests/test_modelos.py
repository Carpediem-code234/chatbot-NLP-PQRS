import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.bot import RESPUESTA_R1, regla_alerta
from src.datos import cargar_todo
from src.evaluacion import curva_umbral, elegir_tau
from src.modelos import m0_mayoritaria, m1_baseline, m3_propuesto
from src.preprocesamiento import normalizar


@pytest.fixture(scope="module")
def D():
    return cargar_todo()


@pytest.fixture(scope="module")
def m3(D):
    return m3_propuesto().fit(D["train"], D["train"].tipo_pqrs)


def f1(m, X):
    return f1_score(X.tipo_pqrs, m.predict(X), average="macro")


def test_modelos_superan_al_piso(D, m3):
    m0 = m0_mayoritaria().fit(D["train"], D["train"].tipo_pqrs)
    m1 = m1_baseline().fit(D["train"], D["train"].tipo_pqrs)
    piso = f1(m0, D["val"])
    assert f1(m1, D["val"]) > piso + 0.3
    assert f1(m3, D["val"]) > piso + 0.3


def test_m3_reproduce_resultado_fase2(D, m3):
    """Regresión: F1-macro de M3 en validación reportado en la Fase 2 (0.657)."""
    assert f1(m3, D["val"]) == pytest.approx(0.809, abs=0.01)


def test_probabilidades_validas(D, m3):
    p = m3.predict_proba(D["test"])
    assert np.allclose(p.sum(1), 1) and (p >= 0).all()


def test_umbral_logra_objetivo_en_validacion(D, m3):
    curva = curva_umbral(m3, D["val"])
    tau = elegir_tau(curva, objetivo=0.9)
    assert curva.loc[curva.tau == tau, "accuracy_decididos"].iloc[0] >= 0.9


@pytest.mark.parametrize("texto", ["mi bebe convulsiona", "no respira", "esta sangrando mucho", "se desmayo"])
def test_regla_urgencia_detecta(texto):
    assert regla_alerta(normalizar(texto))


@pytest.mark.parametrize("texto", ["necesito una cita", "no me entregaron la insulina", "hola"])
def test_regla_urgencia_no_falsa_alarma(texto):
    assert not regla_alerta(normalizar(texto))


def test_respuesta_urgencia_menciona_linea():
    assert "123" in RESPUESTA_R1
