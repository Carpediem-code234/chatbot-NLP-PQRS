import pytest

from app.safety import check_rules


@pytest.mark.parametrize("text", [
    "Se cayó y no se mueve, ayúdenme",
    "mi papá no respira",
    "tengo dolor de pecho muy fuerte",
    "se desmayó mi abuela",
    "esta sangrando mucho",
    "SE ESTÁ AHOGANDO!!",
    "tiene convulsiones",
])
def test_detecta_urgencias(text):
    assert check_rules(text) == "urgencia_medica"


@pytest.mark.parametrize("text", [
    "ayuda",
    "Ayuda eso",
    "ayúdenme con mi cita",
    "mi trámite no se mueve",
    "me hicieron examen de sangre",
    "quiero cancelar mi cita",
    "hola",
])
def test_no_falsas_alarmas(text):
    assert check_rules(text) is None


@pytest.mark.parametrize("text", [
    "agente",
    "quiero hablar con una persona",
    "necesito un asesor",
    "pásame con un humano",
])
def test_solicitud_de_agente(text):
    assert check_rules(text) == "hablar_agente"