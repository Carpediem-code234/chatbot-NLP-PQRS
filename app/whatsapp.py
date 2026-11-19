"""Cliente mínimo de la WhatsApp Cloud API (Meta Graph API)."""
import hashlib
import hmac
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("pqrs-bot")


@dataclass
class MensajeEntrante:
    id: str
    telefono: str
    tipo: str            # text | audio | image | interactive | ...
    texto: str | None


def verificar_firma(cuerpo: bytes, cabecera: str | None, app_secret: str) -> bool:
    """Valida la cabecera X-Hub-Signature-256 = 'sha256=<HMAC-SHA256(cuerpo, APP_SECRET)>'."""
    if not app_secret:
        return True                                   # modo desarrollo (se advierte al arrancar)
    if not cabecera or not cabecera.startswith("sha256="):
        return False
    esperado = hmac.new(app_secret.encode(), cuerpo, hashlib.sha256).hexdigest()
    return hmac.compare_digest(esperado, cabecera.split("=", 1)[1])


def extraer_mensajes(payload: dict) -> list[MensajeEntrante]:
    """Extrae los mensajes de un webhook de WhatsApp. Ignora notificaciones de estado (sent/delivered/read)."""
    salida = []
    for entrada in payload.get("entry", []):
        for cambio in entrada.get("changes", []):
            for msg in cambio.get("value", {}).get("messages", []) or []:
                tipo = msg.get("type", "")
                texto = None
                if tipo == "text":
                    texto = msg.get("text", {}).get("body", "")
                elif tipo == "interactive":                        # respuestas a botones / listas
                    inter = msg.get("interactive", {})
                    texto = (inter.get("button_reply") or inter.get("list_reply") or {}).get("id")
                elif tipo == "button":
                    texto = msg.get("button", {}).get("payload") or msg.get("button", {}).get("text")
                salida.append(MensajeEntrante(msg.get("id", ""), msg.get("from", ""), tipo, texto))
    return salida


class ClienteWhatsApp:
    def __init__(self, token: str, phone_number_id: str, graph_version: str = "v23.0"):
        self.url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"
        self.headers = {"Authorization": f"Bearer {token}"}

    async def enviar_texto(self, telefono: str, cuerpo: str) -> bool:
        payload = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": telefono,
                   "type": "text", "text": {"preview_url": False, "body": cuerpo}}
        try:
            async with httpx.AsyncClient(timeout=15) as cli:
                r = await cli.post(self.url, json=payload, headers=self.headers)
        except httpx.HTTPError as e:
            log.error("No se pudo conectar con WhatsApp: %s", e)
            return False
        if r.status_code >= 400:
            log.error("WhatsApp respondió %s: %s", r.status_code, r.text[:500])
            return False
        return True

    async def marcar_leido(self, msg_id: str):
        payload = {"messaging_product": "whatsapp", "status": "read", "message_id": msg_id}
        try:
            async with httpx.AsyncClient(timeout=10) as cli:
                await cli.post(self.url, json=payload, headers=self.headers)
        except httpx.HTTPError:
            pass
