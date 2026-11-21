"""Lógica de conversación del chatbot (independiente de WhatsApp, para poder probarla y simularla).

Estados de una conversación:
    inicio            → espera un mensaje nuevo
    confirmando       → el modelo propuso un tipo de PQRS con confianza ≥ τ; se pide confirmar (1 Sí / 2 No)
    eligiendo_tipo    → confianza < τ o el usuario dijo que no: menú 1-4 (regla R2)
    pidiendo_documento→ se pide el documento para radicar
    pidiendo_radicado → el usuario quiere saber cómo va su trámite

En CUALQUIER estado se revisan primero las reglas de seguridad (urgencia → 123; agente → funcionario)
y las palabras para cancelar.
"""
import logging
import re

import joblib
import pandas as pd

from app import mensajes as M
from app.almacen import Almacen
from app.formulario_admin import FormularioAdmin
from app.reentrenamiento import Reentrenador
from app.seguridad import revisar
from src.preprocesamiento import cargar_way2esp, normalizar, preparar

log = logging.getLogger("pqrs-bot")

INTENCION_TIPO = {
    "solicitar_cita": "Peticion", "solicitar_autorizacion": "Peticion", "solicitar_documento": "Peticion",
    "solicitar_interprete": "Peticion", "consultar_estado": "Peticion",
    "mal_trato": "Queja", "demora_atencion": "Queja", "infraestructura": "Queja",
    "medicamento_no_entregado": "Reclamo", "cita_incumplida": "Reclamo", "cobro_indebido": "Reclamo",
    "servicio_negado": "Reclamo", "mejorar_atencion": "Sugerencia", "enfoque_intercultural": "Sugerencia",
}
OPCIONES_MENU = {"1": "Peticion", "2": "Queja", "3": "Reclamo", "4": "Sugerencia"}
PALABRAS_MENU = {"pedir": "Peticion", "peticion": "Peticion", "queja": "Queja", "reclamo": "Reclamo",
                 "sugerencia": "Sugerencia", "idea": "Sugerencia"}
SI = {"1", "si", "aa", "ok", "okey", "correcto", "claro", "dale", "listo", "eso", "exacto", "asi es"}
NO = {"2", "no", "nop", "incorrecto", "falso", "esta mal"}
CANCELAR = {"cancelar", "salir", "reiniciar", "menu", "0", "empezar de nuevo"}
RADICADO_RE = re.compile(r"pqrs\W*(\d{8})\W*(\d{1,5})", re.I)


class Modelo:
    """Envuelve el modelo M3 entrenado en la Fase 2 (models/modelo_pqrs_m3.joblib)."""

    def __init__(self, ruta):
        art = joblib.load(ruta)
        self.m_tipo, self.m_int, self.tau = art["tipo"], art["intencion"], float(art["tau"])

    def predecir(self, texto: str) -> dict:
        fila = preparar(pd.DataFrame({"texto": [texto]}))
        p = self.m_tipo.predict_proba(fila)[0]
        return {"tipo": str(self.m_tipo.classes_[p.argmax()]), "confianza": float(p.max()),
                "intencion": str(self.m_int.predict(fila)[0]),
                "probabilidades": {str(c): round(float(v), 3) for c, v in zip(self.m_tipo.classes_, p)}}


def _tiene_way(texto: str) -> bool:
    way = cargar_way2esp()
    return any(w.strip(".,!?¡¿'").lower() in way or w.lower().startswith("jamaya") for w in texto.split())


def _primera_palabra(norm: str) -> str:
    return norm.split()[0] if norm.split() else ""


class Dialogo:
    def __init__(self, almacen: Almacen, modelo: Modelo, minutos_sesion: int = 30, admins=frozenset(),
                 modelo_path=None, notificar=None):
        self.db, self.modelo, self.minutos = almacen, modelo, minutos_sesion
        self.formulario = FormularioAdmin(almacen, set(admins))
        # Reentrenamiento automático tras cada #nuevo (solo si se indica dónde está el modelo y cómo avisar)
        self.modelo_path, self.notificar = modelo_path, notificar
        if modelo_path and notificar:
            self.reentrenador = Reentrenador(self._fin_reentreno)
            self.formulario.al_guardar = self._pedir_reentreno

    def _pedir_reentreno(self, tel):
        if self.reentrenador.lanzar(tel):
            return "⏳ Reentrenando el modelo con el nuevo dato (1–2 min). Te aviso al terminar."
        return "⏳ Ya hay un reentrenamiento en curso; al terminar se hará otro que incluye este dato."

    def _fin_reentreno(self, tel, ok, resumen):
        if ok:
            self.modelo = Modelo(self.modelo_path)                 # recarga en caliente, sin reiniciar
            self.notificar(tel, f"✅ *Modelo actualizado*\n{resumen}")
        else:
            self.notificar(tel, f"⚠️ El reentrenamiento falló; sigo con el modelo anterior.\n{resumen}")

    # ── punto de entrada ─────────────────────────────────────────────────────────
    def responder(self, telefono: str, texto: str) -> list[str]:
        estado, datos = self.db.leer_conversacion(telefono, self.minutos)
        if self.formulario.activa(telefono, texto, estado):      # #nuevo (solo administradores)
            return self.formulario.responder(telefono, texto, estado, datos)
        norm = normalizar(texto)

        regla = revisar(texto)
        if regla == "urgencia":
            self.db.registrar_evento(telefono, "urgencia", texto)
            return [M.URGENCIA]                       # no cambia el estado: puede seguir después
        if regla == "agente":
            self.db.registrar_evento(telefono, "agente", texto)
            self.db.guardar_conversacion(telefono, "inicio")
            return [M.AGENTE]
        if norm in CANCELAR and estado != "inicio":
            self.db.guardar_conversacion(telefono, "inicio")
            return [M.CANCELADO]

        manejador = {"confirmando": self._confirmando, "eligiendo_tipo": self._eligiendo_tipo,
                     "pidiendo_documento": self._pidiendo_documento,
                     "pidiendo_radicado": self._pidiendo_radicado}.get(estado, self._inicio)
        return manejador(telefono, texto, norm, datos)

    # ── estados ──────────────────────────────────────────────────────────────────
    def _inicio(self, tel, texto, norm, datos):
        if not norm:
            return [M.SIN_CONTENIDO]
        m = RADICADO_RE.search(texto)
        if m:
            return self._consultar(tel, f"PQRS-{m.group(1)}-{int(m.group(2)):05d}")

        pred = self.modelo.predecir(texto)
        log.info("modelo: %s", {k: v for k, v in pred.items() if k != "probabilidades"})
        tipo, conf, intencion = pred["tipo"], pred["confianza"], pred["intencion"]

        if conf < self.modelo.tau:                     # regla R2: preguntar en vez de adivinar
            self.db.guardar_conversacion(tel, "eligiendo_tipo", {"descripcion": texto, "intencion": intencion,
                                                                 "confianza": conf})
            return [M.MENU]
        if tipo == "Otro":
            if intencion == "saludo":
                return [M.BIENVENIDA_WAY if _tiene_way(texto) else M.BIENVENIDA]
            if intencion == "agradecimiento":
                return [M.AGRADECIMIENTO]
            if intencion == "sin_contenido":
                return [M.SIN_CONTENIDO]
            return [M.FUERA_DE_ALCANCE]
        if intencion == "consultar_estado":
            self.db.guardar_conversacion(tel, "pidiendo_radicado")
            return [M.PEDIR_RADICADO]

        if INTENCION_TIPO.get(intencion) != tipo:      # intención incoherente con el tipo: no se menciona
            intencion = "sin_especificar"
        self.db.guardar_conversacion(tel, "confirmando", {"descripcion": texto, "tipo": tipo, "intencion": intencion,
                                                          "confianza": conf, "origen": "modelo"})
        sobre = f", sobre *{M.NOMBRE_INTENCION[intencion]}*" if intencion in M.NOMBRE_INTENCION else ""
        saludo = ["¡Jamaya! 👋"] if _tiene_way(texto) and _primera_palabra(norm) in {"hola", "buenos", "jamaya"} else []
        return saludo + [M.CONFIRMAR.format(tipo=M.NOMBRE_TIPO[tipo], sobre=sobre)]

    def _confirmando(self, tel, texto, norm, datos):
        if norm in SI or _primera_palabra(norm) in {"si", "aa"}:
            self.db.guardar_conversacion(tel, "pidiendo_documento", datos)
            return [M.PEDIR_DOCUMENTO]
        if norm in NO or _primera_palabra(norm) == "no":
            datos.update(origen="menu")
            self.db.guardar_conversacion(tel, "eligiendo_tipo", datos)
            return [M.MENU]
        return self._mensaje_nuevo_o_repetir(tel, texto, norm, datos, M.OPCION_INVALIDA)

    def _eligiendo_tipo(self, tel, texto, norm, datos):
        tipo = OPCIONES_MENU.get(norm) or next((t for p, t in PALABRAS_MENU.items() if p in norm.split()), None)
        if not tipo:
            return self._mensaje_nuevo_o_repetir(tel, texto, norm, datos, M.OPCION_INVALIDA + "\n\n" + M.MENU)
        intencion = datos.get("intencion")
        datos.update(tipo=tipo, origen="menu",
                     intencion=intencion if INTENCION_TIPO.get(intencion) == tipo else "sin_especificar")
        self.db.guardar_conversacion(tel, "pidiendo_documento", datos)
        return [M.PEDIR_DOCUMENTO]

    def _pidiendo_documento(self, tel, texto, norm, datos):
        solo_digitos = re.sub(r"[.\s-]", "", texto)
        m = re.search(r"\d{5,12}", solo_digitos)
        if m:
            documento = m.group(0)
        elif "no tengo" in norm or norm in {"no", "ninguno", "nnojo"}:
            documento = None
        else:
            intentos = datos.get("intentos_doc", 0) + 1
            if intentos < 2:
                datos["intentos_doc"] = intentos
                self.db.guardar_conversacion(tel, "pidiendo_documento", datos)
                return [M.DOCUMENTO_INVALIDO]
            documento = None                           # no bloquear al usuario: se radica sin documento
        radicado = self.db.radicar(tel, datos["tipo"], datos.get("intencion"), datos["descripcion"], documento,
                                   datos.get("confianza"), datos.get("origen", "modelo"))
        self.db.guardar_conversacion(tel, "inicio")
        return [M.RADICADO.format(tipo_corto=M.TIPO_CORTO[datos["tipo"]], radicado=radicado)]

    def _pidiendo_radicado(self, tel, texto, norm, datos):
        m = RADICADO_RE.search(texto)
        if not m:
            return self._mensaje_nuevo_o_repetir(tel, texto, norm, datos, M.PEDIR_RADICADO)
        self.db.guardar_conversacion(tel, "inicio")
        return self._consultar(tel, f"PQRS-{m.group(1)}-{int(m.group(2)):05d}")

    # ── utilidades ───────────────────────────────────────────────────────────────
    def _consultar(self, tel, radicado):
        r = self.db.buscar(radicado)
        if not r or r["telefono"] != tel:              # solo el dueño puede consultar su radicado
            return [M.RADICADO_NO_ENCONTRADO.format(radicado=radicado)]
        resp = f"\nRespuesta: {r['respuesta']}" if r.get("respuesta") else ""
        return [M.INFO_RADICADO.format(radicado=r["radicado"], tipo_corto=M.TIPO_CORTO[r["tipo"]],
                                       estado=M.ESTADO_RADICADO[r["estado"]], respuesta=resp)]

    def _mensaje_nuevo_o_repetir(self, tel, texto, norm, datos, repetir):
        """Si en medio de una pregunta el usuario escribe un mensaje largo, se trata como una PQRS nueva."""
        if len(norm.split()) >= 5:
            self.db.guardar_conversacion(tel, "inicio")
            return self._inicio(tel, texto, norm, {})
        self.db.guardar_conversacion(tel, self.db.leer_conversacion(tel, self.minutos)[0], datos)
        return [repetir]
