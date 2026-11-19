"""Persistencia en SQLite: PQRS radicadas, estado de cada conversación y mensajes ya procesados."""
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

ESTADOS_PQRS = ["radicada", "en_tramite", "respondida", "cerrada"]

ESQUEMA = """
CREATE TABLE IF NOT EXISTS pqrs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    radicado    TEXT UNIQUE,
    telefono    TEXT NOT NULL,
    tipo        TEXT NOT NULL,
    intencion   TEXT,
    descripcion TEXT NOT NULL,
    documento   TEXT,
    confianza   REAL,
    origen      TEXT NOT NULL,          -- modelo | menu (el usuario eligió el tipo)
    estado      TEXT NOT NULL DEFAULT 'radicada',
    respuesta   TEXT,
    creado      TEXT NOT NULL,
    actualizado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversaciones (
    telefono    TEXT PRIMARY KEY,
    estado      TEXT NOT NULL,
    datos       TEXT NOT NULL DEFAULT '{}',
    actualizado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mensajes_procesados (
    id          TEXT PRIMARY KEY,
    recibido    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eventos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    telefono    TEXT,
    tipo        TEXT NOT NULL,          -- urgencia | agente | audio | ...
    detalle     TEXT,
    creado      TEXT NOT NULL
);
"""


def ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Almacen:
    def __init__(self, ruta: Path | str):
        self.ruta = str(ruta)
        if self.ruta != ":memory:":
            Path(self.ruta).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.cx = sqlite3.connect(self.ruta, check_same_thread=False)
        self.cx.row_factory = sqlite3.Row
        self.cx.executescript(ESQUEMA)

    # ── deduplicación de mensajes (Meta puede reenviar el mismo webhook) ─────────
    def marcar_procesado(self, msg_id: str) -> bool:
        """True si el mensaje es nuevo; False si ya se había procesado."""
        with self._lock, self.cx:
            try:
                self.cx.execute("INSERT INTO mensajes_procesados VALUES (?, ?)", (msg_id, ahora()))
                return True
            except sqlite3.IntegrityError:
                return False

    def limpiar_procesados(self, dias: int = 7):
        limite = (datetime.now() - timedelta(days=dias)).isoformat(timespec="seconds")
        with self._lock, self.cx:
            self.cx.execute("DELETE FROM mensajes_procesados WHERE recibido < ?", (limite,))

    # ── estado de la conversación ────────────────────────────────────────────────
    def leer_conversacion(self, telefono: str, minutos_sesion: int = 30):
        r = self.cx.execute("SELECT * FROM conversaciones WHERE telefono = ?", (telefono,)).fetchone()
        if not r:
            return "inicio", {}
        if datetime.fromisoformat(r["actualizado"]) < datetime.now() - timedelta(minutes=minutos_sesion):
            return "inicio", {}                      # sesión vencida: se empieza de nuevo
        return r["estado"], json.loads(r["datos"])

    def guardar_conversacion(self, telefono: str, estado: str, datos: dict | None = None):
        with self._lock, self.cx:
            self.cx.execute("INSERT INTO conversaciones VALUES (?, ?, ?, ?) ON CONFLICT(telefono) DO UPDATE SET "
                            "estado = excluded.estado, datos = excluded.datos, actualizado = excluded.actualizado",
                            (telefono, estado, json.dumps(datos or {}, ensure_ascii=False), ahora()))

    # ── PQRS ─────────────────────────────────────────────────────────────────────
    def radicar(self, telefono, tipo, intencion, descripcion, documento, confianza, origen) -> str:
        with self._lock, self.cx:
            cur = self.cx.execute(
                "INSERT INTO pqrs (telefono, tipo, intencion, descripcion, documento, confianza, origen, creado, actualizado)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (telefono, tipo, intencion, descripcion, documento, confianza, origen, ahora(), ahora()))
            radicado = f"PQRS-{datetime.now():%Y%m%d}-{cur.lastrowid:05d}"
            self.cx.execute("UPDATE pqrs SET radicado = ? WHERE id = ?", (radicado, cur.lastrowid))
        return radicado

    def buscar(self, radicado: str):
        r = self.cx.execute("SELECT * FROM pqrs WHERE radicado = ?", (radicado.upper(),)).fetchone()
        return dict(r) if r else None

    def listar(self, estado: str | None = None, tipo: str | None = None, limite: int = 100):
        q, p = "SELECT * FROM pqrs WHERE 1=1", []
        if estado:
            q += " AND estado = ?"; p.append(estado)
        if tipo:
            q += " AND tipo = ?"; p.append(tipo)
        q += " ORDER BY id DESC LIMIT ?"; p.append(limite)
        return [dict(r) for r in self.cx.execute(q, p).fetchall()]

    def actualizar(self, radicado: str, estado: str | None = None, respuesta: str | None = None):
        if estado and estado not in ESTADOS_PQRS:
            raise ValueError(f"estado inválido: {estado}")
        with self._lock, self.cx:
            cur = self.cx.execute("UPDATE pqrs SET estado = COALESCE(?, estado), respuesta = COALESCE(?, respuesta), "
                                  "actualizado = ? WHERE radicado = ?", (estado, respuesta, ahora(), radicado.upper()))
        return cur.rowcount > 0

    # ── eventos (urgencias, solicitudes de agente, audios) ───────────────────────
    def registrar_evento(self, telefono, tipo, detalle=None):
        with self._lock, self.cx:
            self.cx.execute("INSERT INTO eventos (telefono, tipo, detalle, creado) VALUES (?, ?, ?, ?)",
                            (telefono, tipo, detalle, ahora()))

    def listar_eventos(self, tipo: str | None = None, limite: int = 100):
        q, p = "SELECT * FROM eventos", []
        if tipo:
            q += " WHERE tipo = ?"; p.append(tipo)
        q += " ORDER BY id DESC LIMIT ?"; p.append(limite)
        return [dict(r) for r in self.cx.execute(q, p).fetchall()]

    def resumen(self):
        porta = lambda col: {r[0]: r[1] for r in self.cx.execute(f"SELECT {col}, COUNT(*) FROM pqrs GROUP BY {col}")}
        ev = {r[0]: r[1] for r in self.cx.execute("SELECT tipo, COUNT(*) FROM eventos GROUP BY tipo")}
        return {"por_tipo": porta("tipo"), "por_estado": porta("estado"), "eventos": ev}
