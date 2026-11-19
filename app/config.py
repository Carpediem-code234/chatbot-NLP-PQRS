"""Configuración del servicio leída de variables de entorno (archivo .env en la raíz)."""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.config import MODELO, RAIZ

load_dotenv(RAIZ / ".env")
log = logging.getLogger("pqrs-bot")


@dataclass(frozen=True)
class Config:
    # WhatsApp Cloud API (Meta for Developers → tu app → WhatsApp → Configuración de la API)
    wa_token: str = field(default_factory=lambda: os.getenv("WA_TOKEN", "").strip())
    phone_number_id: str = field(default_factory=lambda: os.getenv("PHONE_NUMBER_ID", "").strip())
    verify_token: str = field(default_factory=lambda: os.getenv("VERIFY_TOKEN", "").strip())
    app_secret: str = field(default_factory=lambda: os.getenv("APP_SECRET", "").strip())
    graph_version: str = field(default_factory=lambda: os.getenv("GRAPH_VERSION", "v23.0").strip())
    # Servicio
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", "").strip())
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", str(RAIZ / "data" / "pqrs.db"))))
    modelo_path: Path = field(default_factory=lambda: Path(os.getenv("MODELO_PATH", str(MODELO))))
    minutos_sesion: int = field(default_factory=lambda: int(os.getenv("MINUTOS_SESION", "30")))

    def advertencias(self):
        faltan = [n for n, v in [("WA_TOKEN", self.wa_token), ("PHONE_NUMBER_ID", self.phone_number_id),
                                 ("VERIFY_TOKEN", self.verify_token)] if not v]
        if faltan:
            log.warning("Faltan variables de WhatsApp: %s. El bot no podrá enviar mensajes.", ", ".join(faltan))
        if not self.app_secret:
            log.warning("APP_SECRET vacío: NO se verifica la firma de Meta (solo aceptable en pruebas locales).")
        if not self.admin_token:
            log.warning("ADMIN_TOKEN vacío: los endpoints /api/pqrs quedan deshabilitados.")
