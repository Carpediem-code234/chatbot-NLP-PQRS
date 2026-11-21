"""Formulario #nuevo para agregar registros de entrenamiento por WhatsApp (solo administradores)."""
import pandas as pd
import pytest

from app.almacen import Almacen
from app.dialogo import Dialogo, Modelo
from src import registros as R
from src.config import MODELO

ADMIN = "573136672242"


@pytest.fixture(scope="module")
def modelo():
    return Modelo(MODELO)


@pytest.fixture
def bot(tmp_path, modelo, monkeypatch):
    monkeypatch.setattr(R, "REGISTROS_NUEVOS", tmp_path / "registros_nuevos.csv")
    return Dialogo(Almacen(tmp_path / "t.db"), modelo, admins={ADMIN})


def charla(bot, tel, *mensajes):
    r = []
    for m in mensajes:
        r = bot.responder(tel, m)
    return r


def test_flujo_completo_guarda_registro(bot):
    r = charla(bot, ADMIN, "#nuevo", "mi achonni lleva 3 dias con tuttaa y no le dan sita", "1", "1", "3", "1")
    assert "Guardado" in r[0]
    d = pd.read_csv(R.REGISTROS_NUEVOS, keep_default_na=False)
    fila = d.iloc[0]
    assert (fila.tipo_pqrs, fila.intencion, fila.registro, fila.fuente) == \
        ("Peticion", "solicitar_cita", "simplificado", "manual_whatsapp")
    assert bool(fila.wayuunaiki) is True and fila.plantilla_id.startswith("manual_")


def test_intenciones_dependen_del_tipo(bot):
    r = charla(bot, ADMIN, "#nuevo", "en el baño del puesto de salud no hay agua desde hace semanas", "2")
    assert "Mal trato" in r[0] and "Pedir una cita" not in r[0]


def test_cancelar_no_guarda(bot):
    r = charla(bot, ADMIN, "#nuevo", "me cobraron la consulta siendo del sisben", "cancelar")
    assert "cancelado" in r[0] and not R.REGISTROS_NUEVOS.exists()


def test_rechaza_mensaje_repetido(bot):
    texto = "me cobraron la consulta siendo del sisben y eso no es justo"
    charla(bot, ADMIN, "#nuevo", texto, "3", "3", "2", "1")
    r = charla(bot, ADMIN, "#nuevo", texto)
    assert "ya existe" in r[0]


def test_urgencia_dentro_del_formulario_no_dispara_123(bot):
    r = charla(bot, ADMIN, "#nuevo", "me duele mucho el pecho y no me dan cita con cardiologia")
    assert "TIPO" in r[0]


def test_no_admin_no_puede_usar_formulario(bot):
    r = charla(bot, "573000000000", "#nuevo")
    assert "Nuevo registro" not in r[0]


def test_validar_rechaza_intencion_de_otro_tipo():
    assert "no corresponde" in R.validar("texto de prueba unico xyz", "Queja", "solicitar_cita")


def test_reentrena_al_guardar_y_recarga_modelo(tmp_path, modelo, monkeypatch):
    """Con reentrenamiento activo: al guardar se lanza, y al terminar recarga el modelo y avisa al admin."""
    import app.dialogo as dlg
    monkeypatch.setattr(R, "REGISTROS_NUEVOS", tmp_path / "registros_nuevos.csv")
    avisos, lanzados = [], []

    class FalsoReentrenador:                                  # no entrena de verdad (sería lento)
        def __init__(self, al_terminar):
            self.al_terminar = al_terminar

        def lanzar(self, tel):
            lanzados.append(tel)
            self.al_terminar(tel, True, "10.731 mensajes · F1 externo 0,86 · τ = 0,55")
            return True

    monkeypatch.setattr(dlg, "Reentrenador", FalsoReentrenador)
    bot = Dialogo(Almacen(tmp_path / "t.db"), modelo, admins={ADMIN}, modelo_path=MODELO,
                  notificar=lambda tel, txt: avisos.append(txt))
    r = charla(bot, ADMIN, "#nuevo", "necesito que me den la cita de control prenatal en uribia", "1", "1", "2", "1")
    assert "Guardado" in r[0] and "Reentrenando" in r[1]
    assert lanzados == [ADMIN] and "Modelo actualizado" in avisos[0]


def test_resumen_de_salida_de_entrenamiento():
    from app.reentrenamiento import _resumen
    salida = ("Mensajes sin contenido (regla R0): 30 | limpio: 9,830 | train 5906\n"
              "M3 Propuesto     0.797  0.744    0.812\nUmbral τ elegido en validación: 0.55\n")
    assert _resumen(salida) == "9.830 mensajes · F1 test 0,74 · F1 externo 0,81 · τ = 0,55"
