"""Construye un glosario AMPLIADO wayuunaiki→español a partir del corpus paralelo de data/external/wayuu.

Toma las entradas cortas tipo diccionario del corpus (1 palabra wayuu → 1-2 palabras en español) y las filtra
para no dañar mensajes en español:
  * descarta palabras wayuu que también aparecen como palabras en español (en el corpus o en los PQRS),
  * descarta palabras ambiguas (con más de un significado distinto en el corpus),
  * descarta palabras muy cortas (< 4 letras).
El glosario original (data/raw/glosario_salud_esp_way.csv) siempre tiene prioridad sobre este.

Uso:
    python -m src.glosario_wayuu      # escribe data/raw/glosario_way_esp_ampliado.csv
"""
import re
from collections import Counter, defaultdict

import pandas as pd

from src.config import DATA_RAW, DATASET_PQRS, TEST_EXTERNO, RAIZ

CORPUS = RAIZ / "data" / "external" / "wayuu" / "wayuu_completo_v2.csv"
SALIDA = DATA_RAW / "glosario_way_esp_ampliado.csv"
# Expresiones frecuentes que el filtro descarta por ambigüedad pero son seguras en este contexto
EXTRA = {"anayaawatsuja'a": "gracias", "ekuulu": "comida", "jintui": "niño", "jamayaa": "hola"}
APOS = "'’‘ʼꞌꞋ`´"


def _limpiar(s: str) -> str:
    import unicodedata
    s = s.lower().strip()
    for c in APOS:
        s = s.replace(c, "'")
    s = s.replace("ñ", "\0")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).replace("\0", "ñ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-zñ0-9' ]+", " ", s)
    return " ".join(s.split())


def construir(corpus=CORPUS):
    d = pd.read_csv(corpus, keep_default_na=False, dtype=str)
    d["w"], d["e"] = d.way.map(_limpiar), d.esp.map(_limpiar)

    # Vocabulario español: todo el lado español del corpus + mensajes PQRS (incluye errores de escritura)
    vocab_esp = set()
    for t in d.e:
        vocab_esp.update(t.split())
    for ruta in (DATASET_PQRS, TEST_EXTERNO):
        for t in pd.read_csv(ruta, keep_default_na=False).texto:
            vocab_esp.update(_limpiar(t).split())
    # Las palabras wayuu ya presentes en los PQRS (mensajes mezclados) no cuentan como español
    glos = pd.read_csv(DATA_RAW / "glosario_salud_esp_way.csv", keep_default_na=False)
    vocab_esp -= {_limpiar(w) for w in glos.way}

    cortas = d[(d.w.str.count(" ") == 0) & (d.w.str.len() >= 4) & (d.e.str.count(" ") <= 1) & (d.e != "")]
    significados = defaultdict(Counter)
    for w, e in zip(cortas.w, cortas.e):
        significados[w][e] += 1

    filas = []
    for w, c in significados.items():
        if w in vocab_esp or w.strip("'") != w or len(c) > 1:
            continue
        filas.append({"way": w, "esp": next(iter(c)), "fuente": "corpus_wayuu_completo_v2"})
    for w, e in EXTRA.items():
        if w not in {f["way"] for f in filas}:
            filas.append({"way": w, "esp": e, "fuente": "manual"})
    out = pd.DataFrame(filas).sort_values("way").reset_index(drop=True)
    out.to_csv(SALIDA, index=False)
    return out


if __name__ == "__main__":
    g = construir()
    print(f"{len(g):,} palabras wayuu → español guardadas en {SALIDA}")
