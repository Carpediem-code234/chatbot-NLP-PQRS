"""Clasifica mensajes con el modelo guardado (requiere haber corrido `python -m src.entrenar`).

Uso:
    python -m src.predecir "no me han entregado la insulina"
    python -m src.predecir            # modo interactivo
"""
import json
import sys

from src.bot import ClasificadorPQRS


def main():
    bot = ClasificadorPQRS()
    mensajes = sys.argv[1:]
    if mensajes:
        for t in mensajes:
            print(json.dumps({"mensaje": t, **bot.clasificar(t)}, ensure_ascii=False))
        return
    print("Escribe un mensaje (Ctrl+C para salir)")
    while True:
        try:
            t = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        print(json.dumps(bot.clasificar(t), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
