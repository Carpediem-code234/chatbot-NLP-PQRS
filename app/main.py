"""API del chatbot de PQRS en salud (FastAPI).

Ejecutar:   uvicorn app.main:app --host 0.0.0.0 --port 8000
Documentación interactiva: http://localhost:8000/docs
"""
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app import mensajes as M
from app.almacen import ESTADOS_PQRS, Almacen
from app.config import Config
from app.dialogo import Dialogo, Modelo
from app.seguridad import revisar
from app.whatsapp import ClienteWhatsApp, extraer_mensajes, verificar_firma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pqrs-bot")

DESCRIPCION = """
Chatbot incluyente de **Peticiones, Quejas, Reclamos y Sugerencias (PQRS)** en salud por **WhatsApp Business**.

* `GET/POST /webhook` — integración con WhatsApp Cloud API (Meta).
* `POST /api/clasificar` — clasifica un mensaje con el modelo M3 + reglas de seguridad (sin radicar nada).
* `/api/pqrs` — consulta y gestión de las PQRS radicadas (requiere cabecera `X-Admin-Token`).
"""

def _notificador(app):
    """Permite enviar un WhatsApp desde el hilo del reentrenamiento (que corre fuera del event loop)."""
    loop = asyncio.get_running_loop()

    def notificar(tel, texto):
        asyncio.run_coroutine_threadsafe(app.state.wa.enviar_texto(tel, texto), loop)
    return notificar

def crear_app(config: Config | None = None, cliente=None) -> FastAPI:
    cfg = config or Config()

    @asynccontextmanager
    async def vida(app: FastAPI):
        cfg.advertencias()
        if not cfg.modelo_path.exists():
            raise RuntimeError(f"No existe el modelo {cfg.modelo_path}. Ejecuta primero: python -m src.entrenar")
        app.state.db = Almacen(cfg.db_path)
        app.state.db.limpiar_procesados()
        app.state.modelo = Modelo(cfg.modelo_path)
        app.state.dialogo = Dialogo(app.state.db, app.state.modelo, cfg.minutos_sesion, cfg.admins_wa,
            modelo_path=cfg.modelo_path, notificar=_notificador(app))
        app.state.wa = cliente or ClienteWhatsApp(cfg.wa_token, cfg.phone_number_id, cfg.graph_version)
        log.info("Bot listo · modelo %s · τ=%.2f · BD %s", cfg.modelo_path.name, app.state.modelo.tau, cfg.db_path)
        yield

    app = FastAPI(title="Chatbot PQRS Salud · WhatsApp", version="1.0.0", description=DESCRIPCION, lifespan=vida)

    def admin(x_admin_token: str = Header(..., description="Valor de ADMIN_TOKEN del archivo .env")):
        if not cfg.admin_token or x_admin_token != cfg.admin_token:
            raise HTTPException(status_code=401, detail="Token de administrador inválido o no configurado")

    # ── WhatsApp ─────────────────────────────────────────────────────────────────
    @app.get("/webhook", response_class=PlainTextResponse, tags=["WhatsApp"],
             summary="Verificación del webhook (Meta la llama una vez al configurarlo)")
    async def verificar(modo: str = Query(None, alias="hub.mode"), token: str = Query(None, alias="hub.verify_token"),
                        reto: str = Query(None, alias="hub.challenge")):
        if modo == "subscribe" and cfg.verify_token and token == cfg.verify_token:
            return reto
        raise HTTPException(status_code=403, detail="verify_token incorrecto")

    async def atender(app_, msg):
        wa, db, dialogo = app_.state.wa, app_.state.db, app_.state.dialogo
        try:
            await wa.marcar_leido(msg.id)
            if msg.tipo == "audio":
                db.registrar_evento(msg.telefono, "audio")
                respuestas = [M.AUDIO]
            elif msg.texto is None:
                respuestas = [M.NO_TEXTO]
            else:
                respuestas = dialogo.responder(msg.telefono, msg.texto)
            for r in respuestas:
                await wa.enviar_texto(msg.telefono, r)
            log.info("%s · %r → %d respuesta(s)", msg.telefono[-4:].rjust(10, "*"), (msg.texto or msg.tipo)[:60],
                     len(respuestas))
        except Exception:
            log.exception("Error atendiendo el mensaje %s", msg.id)

    @app.post("/webhook", tags=["WhatsApp"], summary="Recepción de mensajes de WhatsApp")
    async def recibir(request: Request, tareas: BackgroundTasks,
                      x_hub_signature_256: str | None = Header(None)):
        cuerpo = await request.body()
        if not verificar_firma(cuerpo, x_hub_signature_256, cfg.app_secret):
            log.warning("Firma inválida: petición rechazada")
            raise HTTPException(status_code=401, detail="firma inválida")
        try:
            payload = json.loads(cuerpo or b"{}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="JSON inválido")
        for msg in extraer_mensajes(payload):
            if msg.id and request.app.state.db.marcar_procesado(msg.id):
                tareas.add_task(atender, request.app, msg)   # se responde 200 de inmediato a Meta
        return {"status": "ok"}

    # ── API ──────────────────────────────────────────────────────────────────────
    class Entrada(BaseModel):
        texto: str = Field(..., examples=["no me han entregado la insulina de mi mamá"])

    @app.post("/api/clasificar", tags=["Modelo"], summary="Clasificar un mensaje (no radica)")
    async def clasificar(e: Entrada, request: Request):
        regla = revisar(e.texto)
        if regla:
            return {"accion": f"regla_{regla}"}
        pred = request.app.state.modelo.predecir(e.texto)
        pred["accion"] = "preguntar_menu" if pred["confianza"] < request.app.state.modelo.tau else "confirmar_tipo"
        pred["tau"] = request.app.state.modelo.tau
        return pred

    @app.get("/api/pqrs", tags=["Administración"], dependencies=[Depends(admin)], summary="Listar PQRS radicadas")
    async def listar(request: Request, estado: Literal[tuple(ESTADOS_PQRS)] | None = None,
                     tipo: Literal["Peticion", "Queja", "Reclamo", "Sugerencia"] | None = None,
                     limite: int = Query(100, le=1000)):
        return request.app.state.db.listar(estado, tipo, limite)

    @app.get("/api/pqrs/resumen", tags=["Administración"], dependencies=[Depends(admin)],
             summary="Conteos por tipo, estado y eventos (urgencias, agente, audios)")
    async def resumen(request: Request):
        return request.app.state.db.resumen()

    @app.get("/api/eventos", tags=["Administración"], dependencies=[Depends(admin)],
             summary="Urgencias detectadas, solicitudes de agente y notas de voz recibidas")
    async def eventos(request: Request, tipo: Literal["urgencia", "agente", "audio"] | None = None):
        return request.app.state.db.listar_eventos(tipo)

    @app.get("/api/pqrs/{radicado}", tags=["Administración"], dependencies=[Depends(admin)],
             summary="Consultar una PQRS")
    async def obtener(radicado: str, request: Request):
        r = request.app.state.db.buscar(radicado)
        if not r:
            raise HTTPException(status_code=404, detail="radicado no encontrado")
        return r

    class Cambio(BaseModel):
        estado: Literal[tuple(ESTADOS_PQRS)] | None = None
        respuesta: str | None = Field(None, description="Respuesta para el usuario; se le envía por WhatsApp")

    @app.patch("/api/pqrs/{radicado}", tags=["Administración"], dependencies=[Depends(admin)],
               summary="Actualizar estado / responder una PQRS (notifica al usuario por WhatsApp)")
    async def actualizar(radicado: str, c: Cambio, request: Request):
        db = request.app.state.db
        if not db.actualizar(radicado, c.estado, c.respuesta):
            raise HTTPException(status_code=404, detail="radicado no encontrado")
        r = db.buscar(radicado)
        aviso = (f"📬 Novedad en tu radicado *{r['radicado']}*:\nEstado: *{M.ESTADO_RADICADO[r['estado']]}*"
                 + (f"\n{c.respuesta}" if c.respuesta else ""))
        r["notificado"] = await request.app.state.wa.enviar_texto(r["telefono"], aviso)
        return r

    @app.get("/health", tags=["Servicio"])
    async def salud():
        return {"ok": True}

    return app


app = crear_app()
