import pandas as pd
import pytest

from src.preprocesamiento import COLS_META, meta_features, normalizar, preparar, quitar_tildes


@pytest.mark.parametrize("entrada, esperado", [
    ("KIERO SITA PAL DOTOR", "quiero cita para el doctor"),
    ("me reprogramaron la sita 3 veces!!", "me reprogramaron la cita 3 veces"),
    ("Señores, ¿en qué va el trámite?", "señores en que va el tramite"),
    ("en ke va el tramite ke radike", "en que va el tramite que radique"),
    ("no me dan la medisina pa la presion", "no me dan la medicina para la presion"),
])
def test_normaliza_ortografia_y_abreviaturas(entrada, esperado):
    assert normalizar(entrada) == esperado


def test_traduce_wayuunaiki_con_glosario():
    assert normalizar("wattachonya no hay tottoolu") == "buenos dias no hay medico"
    assert normalizar("mi hijo ayuulii") == "mi hijo enfermo"


def test_sin_traduccion_conserva_wayuunaiki():
    assert "tottoolu" in normalizar("no hay tottoolu", traducir_way=False)


def test_unifica_saltillo():
    assert normalizar("e’iijaa") == normalizar("e'iijaa") == "diarrea"


def test_conserva_enie():
    assert quitar_tildes("niño pequeño médico") == "niño pequeño medico"


@pytest.mark.parametrize("texto", ["", "   ", "👍", "🙏🙏", "...", "?"])
def test_mensajes_sin_contenido_quedan_vacios(texto):
    assert normalizar(texto) == ""


def test_meta_features():
    f = meta_features(["NECESITO CITA", "¿hay medico? 🙏", "no hay tottoolu", "no hay medico"]).iloc
    assert list(meta_features(["x"]).columns) == COLS_META
    assert f[0]["prop_mayusculas"] == 1.0
    assert f[1]["tiene_pregunta"] == 1.0 and f[1]["tiene_emoji"] == 1.0
    assert f[2]["tiene_way"] == 1.0 and f[3]["tiene_way"] == 0.0


def test_preparar_agrega_columnas():
    d = preparar(pd.DataFrame({"texto": ["hola", "👍"]}))
    for c in ["texto_norm", "texto_norm_sin_way", "n_palabras", *COLS_META]:
        assert c in d.columns
    assert d.n_palabras.tolist() == [1, 0]
