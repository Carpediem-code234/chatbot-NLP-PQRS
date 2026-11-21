"""Limpieza, normalización y rasgos estructurales de los mensajes de WhatsApp."""
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import GLOSARIO, GLOSARIO_AMPLIADO

APOS = "'’‘ʼꞌꞋ`´"
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

ABREVIATURAS = {"q": "que", "ke": "que", "k": "que", "xq": "porque", "pq": "porque", "xfa": "por favor",
                "porfa": "por favor", "pa": "para", "pal": "para el", "tb": "tambien", "d": "de", "x": "por"}

# Errores frecuentes en escritura con baja alfabetización / teclado del celular
CORRECCIONES = {"sita": "cita", "sitas": "citas", "nesesito": "necesito", "nesecito": "necesito", "medisina": "medicina",
                "medisinas": "medicinas", "ase": "hace", "ay": "hay", "yego": "llego", "yevo": "llevo", "yamaron": "llamaron",
                "canselaron": "cancelaron", "urgensias": "urgencias", "istoria": "historia", "solisito": "solicito",
                "autorizasion": "autorizacion", "atension": "atencion", "esamen": "examen", "dotor": "doctor",
                "kiero": "quiero", "aki": "aqui", "aser": "hacer", "bacunas": "vacunas", "bacunacion": "vacunacion",
                "resibir": "recibir", "nesesitamos": "necesitamos", "consiencia": "conciencia"}

COLS_META = ["log_n_palabras", "prop_mayusculas", "tiene_pregunta", "tiene_emoji", "tiene_numero", "tiene_way"]


@lru_cache(maxsize=None)
def cargar_way2esp(ruta=str(GLOSARIO)):
    """Invierte el glosario esp→way de la Fase 1 en un mapa wayuunaiki→español (solo entradas de 1 palabra)."""
    glos = pd.read_csv(ruta, keep_default_na=False)
    mapa = {}
    for esp, way in zip(glos.termino_esp, glos.way):
        w = way.lower().strip()
        for c in APOS:
            w = w.replace(c, "'")
        w = quitar_tildes(w)                     # 'wanülü' → 'wanulu', igual que el texto normalizado
        if " " not in w and len(w) > 2:          # evita formas muy ambiguas como 'aa' (sí)
            mapa.setdefault(w, esp.replace("tener ", ""))
    # Glosario ampliado desde el corpus paralelo wayuu (si existe); el glosario original tiene prioridad
    if GLOSARIO_AMPLIADO.exists():
        amp = pd.read_csv(GLOSARIO_AMPLIADO, keep_default_na=False)
        for w, esp in zip(amp.way, amp.esp):
            mapa.setdefault(w, esp)
    return mapa


def quitar_tildes(s: str) -> str:
    """Quita tildes conservando la ñ."""
    s = s.replace("ñ", "\0")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.replace("\0", "ñ")


def normalizar(texto: str, traducir_way: bool = True, corregir: bool = True) -> str:
    """Minúsculas, sin tildes, saltillo unificado, sin emojis/puntuación, abreviaturas y errores corregidos,
    y palabras en wayuunaiki traducidas al español con el glosario."""
    way2esp = cargar_way2esp()
    s = texto.lower()
    for c in APOS:
        s = s.replace(c, "'")
    s = EMOJI_RE.sub(" ", s)
    s = quitar_tildes(s)
    s = re.sub(r"[^a-zñü0-9' ]+", " ", s)
    toks = []
    for t in s.split():
        if traducir_way and t in way2esp:
            toks.extend(way2esp[t].split())
            continue
        if corregir:
            t = ABREVIATURAS.get(t, t)
            t = CORRECCIONES.get(t, t)
            if re.fullmatch(r"\w*k[ei]\w*", t) and t not in way2esp:     # radike → radique
                t = re.sub(r"k(?=[ei])", "qu", t)
        toks.append(t)
    return " ".join(" ".join(toks).split())


def meta_features(textos) -> pd.DataFrame:
    """Rasgos del mensaje ORIGINAL que la normalización borra."""
    way2esp = cargar_way2esp()
    filas = []
    for t in textos:
        palabras = re.findall(r"[^\W\d_]+", t)
        letras = [c for c in t if c.isalpha()]
        filas.append({
            "log_n_palabras": np.log1p(len(palabras)),
            "prop_mayusculas": sum(c.isupper() for c in letras) / max(len(letras), 1),
            "tiene_pregunta": float("?" in t or t.lower().startswith(("como", "cuando", "donde", "que", "cual", "en que", "hay"))),
            "tiene_emoji": float(bool(EMOJI_RE.search(t))),
            "tiene_numero": float(bool(re.search(r"\d", t))),
            "tiene_way": float(any(w.strip(APOS).lower() in way2esp for w in t.split())),
        })
    return pd.DataFrame(filas)


def preparar(d: pd.DataFrame) -> pd.DataFrame:
    """Añade texto normalizado, nº de palabras y rasgos estructurales a un DataFrame con columna `texto`."""
    d = d.copy()
    d["texto_norm"] = d.texto.map(normalizar)
    d["texto_norm_sin_way"] = d.texto.map(lambda s: normalizar(s, traducir_way=False))
    d["n_palabras"] = d.texto_norm.str.split().str.len().fillna(0).astype(int)
    return d.join(meta_features(d.texto).set_index(d.index))
