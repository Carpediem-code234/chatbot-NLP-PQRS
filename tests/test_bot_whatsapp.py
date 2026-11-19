"""Pruebas de la Fase 3: reglas de seguridad, diálogo y API/webhook de WhatsApp (sin conectarse a Meta)."""
import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import mensajes as M
from app.almacen import Almacen
from app.config import Config
from app.dialogo import Dialogo, Modelo
from app.main import crear_app
from app.seguridad import revisar
from app.whatsapp import extraer_mensajes, verificar_firma
from src.config import MODELO

pytestmark = pytest.mark.skipif(not MODELO.exists(), reason="Falta el modelo: ejecuta python -m src.entrenar")


# ── reglas de seguridad (casos de la rama Whatsapp-Api-integracion + nuevos) ──────
@pytest.mark.parametrize("texto", ["Se cayó y no se mueve, ayúdenme", "mi papá no respira", "tengo dolor de pecho muy fuerte",
                                   "se desmayó mi abuela", "esta sangrando mucho", "SE ESTÁ AHOGANDO!!",
                                   "tiene convulsiones", "mi bebe convulsiona", "lo mordio una culebra",
                                   "SE ESTA AOGANDO", "se desmallo", "tiene una emorragia", "el niño esta conbulsionando"])
def test_detecta_urgencias(texto):
    assert revisar(texto) == "urgencia"


@pytest.mark.parametrize("texto", ["ayuda", "Ayuda eso", "ayúdenme con mi cita", "mi trámite no se mueve",
                                   "me hicieron examen de sangre", "quiero cancelar mi cita", "hola",
                                   "no me entregaron la insulina"])
def test_no_falsas_alarmas(texto):
    assert revisar(texto) is None

@pytest.mark.parametrize("texto", [
    "me duele mucho el pecho", "me quiero matar", "ya no quiero vivir", "quiero matarme",
    "se tomó veneno", "se tomo el gramoxone", "me estoy desangrando", "mi esposa está de parto y sangra",
    "está embarazada y está sangrando", "mi bebé se puso morado", "me picó un alacrán",
    "mi hijo se quemó con agua hirviendo", "tuve un accidente de moto", "mi papá está muy grave"])
def test_detecta_urgencias_nuevas(texto):
    assert revisar(texto) == "urgencia"


@pytest.mark.parametrize("texto", [
    "necesito cita de control prenatal", "me cobraron la consulta", "el médico fue grosero",
    "la eps me negó la orden"])
def test_no_falsas_alarmas_nuevas(texto):
    assert revisar(texto) is None

@pytest.mark.parametrize("texto", ["agente", "quiero hablar con una persona", "necesito un asesor",
                                   "pásame con un humano", "kiero una persona de verdad"])
def test_solicitud_de_agente(texto):
    assert revisar(texto) == "agente"


# ── diálogo ───────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def modelo():
    return Modelo(MODELO)


@pytest.fixture
def bot(tmp_path, modelo):
    return Dialogo(Almacen(tmp_path / "t.db"), modelo)


def charla(bot, tel, *mensajes):
    out = []
    for m in mensajes:
        out = bot.responder(tel, m)
    return out


def test_flujo_completo_reclamo(bot):
    r = charla(bot, "57300", "no me han entregado la insulina de mi mama hace 15 dias")
    assert "*reclamo*" in r[-1] and "medicamento" in r[-1]
    assert charla(bot, "57300", "si") == [M.PEDIR_DOCUMENTO]
    r = charla(bot, "57300", "1.234.567")
    assert "PQRS-" in r[0]
    p = bot.db.listar()[0]
    assert (p["tipo"], p["intencion"], p["documento"], p["origen"]) == ("Reclamo", "medicamento_no_entregado", "1234567", "modelo")


def test_baja_confianza_muestra_menu_y_radica_por_menu(bot):
    assert charla(bot, "1", "lo mismo de antes") == [M.MENU]
    charla(bot, "1", "4", "no tengo")
    p = bot.db.listar()[0]
    assert (p["tipo"], p["origen"], p["documento"]) == ("Sugerencia", "menu", None)


def test_usuario_corrige_el_tipo(bot):
    charla(bot, "2", "deberian poner medicos que hablen wayuunaiki")
    assert charla(bot, "2", "2") == [M.MENU]
    charla(bot, "2", "3", "12345678")
    assert bot.db.listar()[0]["tipo"] == "Reclamo"


def test_wayuunaiki_saludo_y_si(bot):
    r = charla(bot, "3", "jamaya, mi hijo ayuulii y no hay tottoolu en el puesto")
    assert r[0].startswith("¡Jamaya!")
    assert charla(bot, "3", "aa") == [M.PEDIR_DOCUMENTO]          # 'aa' = sí en wayuunaiki


def test_urgencia_en_medio_del_flujo_no_pierde_el_estado(bot):
    charla(bot, "4", "no me han entregado la insulina")
    assert charla(bot, "4", "mi bebe no respira") == [M.URGENCIA]
    assert charla(bot, "4", "si") == [M.PEDIR_DOCUMENTO]
    assert bot.db.listar_eventos("urgencia")


def test_cancelar(bot):
    charla(bot, "5", "lo de la otra vez")
    assert charla(bot, "5", "cancelar") == [M.CANCELADO]
    assert bot.db.leer_conversacion("5")[0] == "inicio"


def test_consulta_de_radicado_solo_el_dueno(bot):
    charla(bot, "6", "me cobraron la consulta y soy subsidiado", "si", "no tengo")
    rad = bot.db.listar()[0]["radicado"]
    bot.db.actualizar(rad, "respondida", "Le devolveremos el dinero.")
    r = charla(bot, "6", f"como va el {rad}")
    assert "Respondida" in r[0] and "devolveremos" in r[0]
    assert "No encontré" in charla(bot, "otro", f"como va el {rad}")[0]


def test_mensajes_sin_contenido_y_fuera_de_alcance(bot):
    assert charla(bot, "7", "👍") == [M.SIN_CONTENIDO]
    assert charla(bot, "7", "gracias") == [M.AGRADECIMIENTO]


def test_documento_invalido_no_bloquea(bot):
    charla(bot, "8", "no me han entregado la insulina", "si")
    assert charla(bot, "8", "abc") == [M.DOCUMENTO_INVALIDO]
    assert "PQRS-" in charla(bot, "8", "xyz")[0]


# ── webhook y API ─────────────────────────────────────────────────────────────────
class FalsoWhatsApp:
    def __init__(self):
        self.enviados, self.leidos = [], []

    async def enviar_texto(self, tel, cuerpo):
        self.enviados.append((tel, cuerpo)); return True

    async def marcar_leido(self, msg_id):
        self.leidos.append(msg_id)


SECRETO = "secreto-de-prueba"


@pytest.fixture
def cliente(tmp_path):
    cfg = Config(wa_token="x", phone_number_id="1", verify_token="tok", app_secret=SECRETO, admin_token="adm",
                 db_path=tmp_path / "api.db", modelo_path=MODELO)
    wa = FalsoWhatsApp()
    with TestClient(crear_app(cfg, wa)) as c:
        c.wa = wa
        yield c


def webhook(mensajes):
    return {"object": "whatsapp_business_account", "entry": [{"id": "1", "changes": [{"field": "messages", "value": {
        "messaging_product": "whatsapp", "metadata": {"phone_number_id": "1"}, "messages": mensajes}}]}]}


def msg_texto(i, texto, tel="573001112233"):
    return {"from": tel, "id": f"wamid.{i}", "timestamp": "1", "type": "text", "text": {"body": texto}}


def post_firmado(c, payload):
    cuerpo = json.dumps(payload).encode()
    firma = "sha256=" + hmac.new(SECRETO.encode(), cuerpo, hashlib.sha256).hexdigest()
    return c.post("/webhook", content=cuerpo, headers={"X-Hub-Signature-256": firma, "Content-Type": "application/json"})


def test_verificacion_webhook(cliente):
    ok = cliente.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "tok", "hub.challenge": "123"})
    assert ok.status_code == 200 and ok.text == "123"
    assert cliente.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "mal",
                                           "hub.challenge": "1"}).status_code == 403


def test_firma_invalida_rechazada(cliente):
    r = cliente.post("/webhook", json=webhook([msg_texto(1, "hola")]), headers={"X-Hub-Signature-256": "sha256=00"})
    assert r.status_code == 401 and not cliente.wa.enviados


def test_conversacion_por_webhook_y_deduplicacion(cliente):
    assert post_firmado(cliente, webhook([msg_texto(1, "no me han entregado la insulina")])).status_code == 200
    post_firmado(cliente, webhook([msg_texto(1, "no me han entregado la insulina")]))   # reenvío de Meta
    assert len(cliente.wa.enviados) == 1 and "reclamo" in cliente.wa.enviados[0][1]
    post_firmado(cliente, webhook([msg_texto(2, "si")]))
    post_firmado(cliente, webhook([msg_texto(3, "12345678")]))
    assert "PQRS-" in cliente.wa.enviados[-1][1]
    assert cliente.wa.leidos == ["wamid.1", "wamid.2", "wamid.3"]


def test_audio_y_estados_de_entrega(cliente):
    audio = {"from": "57", "id": "wamid.a", "type": "audio", "audio": {"id": "x"}}
    post_firmado(cliente, webhook([audio]))
    assert cliente.wa.enviados[-1][1] == M.AUDIO
    estados = {"entry": [{"changes": [{"value": {"statuses": [{"id": "wamid.1", "status": "read"}]}}]}]}
    n = len(cliente.wa.enviados)
    assert post_firmado(cliente, estados).status_code == 200 and len(cliente.wa.enviados) == n


def test_api_clasificar(cliente):
    r = cliente.post("/api/clasificar", json={"texto": "no me han entregado la insulina"}).json()
    assert r["tipo"] == "Reclamo" and 0 <= r["confianza"] <= 1
    assert cliente.post("/api/clasificar", json={"texto": "no respira"}).json() == {"accion": "regla_urgencia"}


def test_api_admin(cliente):
    assert cliente.get("/api/pqrs").status_code == 422                        # falta cabecera
    assert cliente.get("/api/pqrs", headers={"X-Admin-Token": "mal"}).status_code == 401
    for i, t in enumerate(["no me han entregado la insulina", "si", "123456"]):
        post_firmado(cliente, webhook([msg_texto(10 + i, t)]))
    h = {"X-Admin-Token": "adm"}
    lista = cliente.get("/api/pqrs", headers=h).json()
    assert len(lista) == 1
    rad = lista[0]["radicado"]
    r = cliente.patch(f"/api/pqrs/{rad}", headers=h, json={"estado": "respondida", "respuesta": "Ya puede reclamarla."})
    assert r.status_code == 200 and r.json()["estado"] == "respondida" and r.json()["notificado"]
    assert "Ya puede reclamarla" in cliente.wa.enviados[-1][1]
    assert cliente.get("/api/pqrs/resumen", headers=h).json()["por_estado"] == {"respondida": 1}
    assert cliente.patch(f"/api/pqrs/{rad}", headers=h, json={"estado": "inventado"}).status_code == 422


def test_extraer_respuesta_de_boton():
    p = webhook([{"from": "57", "id": "w", "type": "interactive",
                  "interactive": {"type": "button_reply", "button_reply": {"id": "1", "title": "Sí"}}}])
    assert extraer_mensajes(p)[0].texto == "1"


def test_firma_sin_secreto_modo_desarrollo():
    assert verificar_firma(b"{}", None, "") is True
    assert verificar_firma(b"{}", None, "s") is False
