import logging
import os
from pathlib import Path

import httpx
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from app.responses import RESPONSES, FALLBACK
from app.safety import check_rules

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pqrs-bot")

VERIFY_TOKEN = os.environ["VERIFY_TOKEN"].strip()
WA_TOKEN = os.environ["WA_TOKEN"].strip()
PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"].strip()
GRAPH_VERSION = os.getenv("GRAPH_VERSION", "v21.0")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.40"))

MODEL = joblib.load(ROOT / "models/baseline.joblib")
app = FastAPI()
seen_ids: set[str] = set()


def predict(text: str) -> tuple[str, str]:
    """Devuelve (intención, origen). Las reglas de seguridad van primero."""
    rule = check_rules(text)
    if rule:
        return rule, "regla"
    proba = MODEL.predict_proba([text])[0]
    best = proba.argmax()
    intent = MODEL.classes_[best]
    if intent != "urgencia_medica" and proba[best] < CONF_THRESHOLD:
        return "no_entendido", "modelo"
    return intent, "modelo"


async def send_text(to: str, body: str) -> None:
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            url, json=payload, headers={"Authorization": f"Bearer {WA_TOKEN}"}
        )
    if r.status_code >= 400:
        log.error("Error al enviar a WhatsApp: %s %s", r.status_code, r.text)
    else:
        log.info("Respuesta enviada a %s", to)


@app.get("/webhook")
async def verify(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403)


@app.post("/webhook")
async def receive(request: Request):
    data = await request.json()
    try:
        value = data["entry"][0]["changes"][0]["value"]
        for msg in value.get("messages", []):
            if msg["id"] in seen_ids:
                continue
            seen_ids.add(msg["id"])
            if msg.get("type") != "text":
                await send_text(msg["from"], FALLBACK)
                continue
            text = msg["text"]["body"]
            intent, source = predict(text)
            log.info("Mensaje: %r -> intención: %s (%s)", text, intent, source)
            await send_text(msg["from"], RESPONSES.get(intent, FALLBACK))
    except (KeyError, IndexError):
        pass
    except Exception:
        log.exception("Error procesando el mensaje")
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"ok": True}