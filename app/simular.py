"""Simula una conversación de WhatsApp en la terminal, sin Meta (usa una base de datos temporal).

Uso:  python -m app.simular
"""
import tempfile
from pathlib import Path

from app.almacen import Almacen
from app.config import Config
from app.dialogo import Dialogo, Modelo


def main():
    cfg = Config()
    db = Almacen(Path(tempfile.mkdtemp()) / "simulacion.db")
    bot = Dialogo(db, Modelo(cfg.modelo_path))
    print("Simulador del chatbot PQRS (Ctrl+C para salir). Escribe como lo harías por WhatsApp.\n")
    while True:
        try:
            texto = input("👤 ")
        except (KeyboardInterrupt, EOFError):
            break
        for r in bot.responder("573000000000", texto):
            print("🤖 " + r.replace("\n", "\n   ") + "\n")
    print("\nPQRS radicadas en esta simulación:")
    for p in db.listar():
        print(f"  {p['radicado']} · {p['tipo']} · {p['intencion']} · doc={p['documento']} · {p['descripcion'][:50]}")


if __name__ == "__main__":
    main()
