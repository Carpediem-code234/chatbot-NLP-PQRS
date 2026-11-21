"""Reentrenamiento automático en segundo plano (se lanza al guardar un registro con #nuevo).

Ejecuta `python -m src.entrenar --rapido` en un hilo aparte, así el bot sigue atendiendo con el modelo anterior.
Al terminar llama a `al_terminar(telefono, ok, resumen)`. Si llegan registros nuevos mientras entrena,
al acabar vuelve a entrenar una vez más para incluirlos.
"""
import logging
import os
import re
import subprocess
import sys
import threading

from src.config import RAIZ

log = logging.getLogger("pqrs-bot")


def _resumen(salida: str) -> str:
    """Extrae de la salida de src.entrenar el F1 de M3 y el umbral τ."""
    m3 = re.search(r"M3 Propuesto\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", salida)
    tau = re.search(r"Umbral .*?:\s*([\d.]+)", salida)
    n = re.search(r"limpio:\s*([\d,.]+)", salida)
    partes = []
    if n:
        partes.append(f"{n.group(1).replace(',', '.')} mensajes")
    if m3:
        partes.append(f"F1 test {float(m3.group(2)):.2f} · F1 externo {float(m3.group(3)):.2f}".replace(".", ","))
    if tau:
        partes.append(f"τ = {tau.group(1).replace('.', ',')}")
    return " · ".join(partes)


class Reentrenador:
    def __init__(self, al_terminar, timeout_s: int = 1200):
        self.al_terminar, self.timeout_s = al_terminar, timeout_s
        self._lock = threading.Lock()
        self._corriendo = False
        self._pendiente = None                       # teléfono que pidió otra vuelta mientras entrenaba

    @property
    def corriendo(self) -> bool:
        return self._corriendo

    def lanzar(self, telefono: str) -> bool:
        """Inicia el reentrenamiento. Devuelve False si ya había uno en curso (queda en cola una vuelta más)."""
        with self._lock:
            if self._corriendo:
                self._pendiente = telefono
                return False
            self._corriendo = True
        threading.Thread(target=self._entrenar, args=(telefono,), daemon=True).start()
        return True

    def _entrenar(self, telefono):
        while True:
            log.info("Reentrenamiento iniciado (pedido por ****%s)", telefono[-4:])
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}      # τ y tildes en la consola de Windows
            try:
                p = subprocess.run([sys.executable, "-m", "src.entrenar", "--rapido"], cwd=RAIZ, env=env,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=self.timeout_s)
                ok, salida = p.returncode == 0, p.stdout + p.stderr
            except Exception as e:                                   # timeout u otro error
                ok, salida = False, str(e)
            resumen = _resumen(salida) if ok else salida.strip().splitlines()[-1][:300] if salida.strip() else "error"
            log.info("Reentrenamiento %s: %s", "OK" if ok else "FALLÓ", resumen)
            try:
                self.al_terminar(telefono, ok, resumen)
            except Exception:
                log.exception("Error al notificar el fin del reentrenamiento")
            with self._lock:
                if self._pendiente is None:
                    self._corriendo = False
                    return
                telefono, self._pendiente = self._pendiente, None
