"""Carga, filtrado, deduplicación y partición train/val/test agrupada por plantilla."""
import numpy as np
import pandas as pd

from src.config import DATASET_PQRS, SEED, TEST_EXTERNO
from src.preprocesamiento import preparar


def cargar_dataset(ruta=DATASET_PQRS) -> pd.DataFrame:
    return pd.read_csv(ruta, keep_default_na=False, dtype={"texto": str})


def limpiar(data: pd.DataFrame):
    """Aparta los mensajes sin palabras (regla R0) y deduplica por texto normalizado + etiqueta.

    Devuelve (dataset_limpio, mensajes_sin_contenido)."""
    sin_contenido = data[data.n_palabras == 0]
    d = data[data.n_palabras > 0].drop_duplicates(["texto_norm", "tipo_pqrs"]).reset_index(drop=True)
    return d, sin_contenido


def particion_por_plantilla(d: pd.DataFrame, seed: int = SEED, con_val: bool = True) -> pd.Series:
    """Para cada intención, reparte sus plantillas: 1 a test, 1 a val (si con_val) y el resto a train.

    Ninguna plantilla aparece en dos particiones (evita la fuga detectada en la Fase 1)."""
    rng = np.random.RandomState(seed)
    asignacion = {}
    for _, g in d.groupby("intencion"):
        plantillas = sorted(g.plantilla_id.unique())
        rng.shuffle(plantillas)
        n_te = max(1, round(.2 * len(plantillas)))
        n_va = max(1, round(.2 * len(plantillas))) if con_val else 0
        for i, p in enumerate(plantillas):
            if i < n_te:
                asignacion[p] = "test"
            elif i < n_te + n_va:
                asignacion[p] = "val"
            else:
                asignacion[p] = "train"
    return d.plantilla_id.map(asignacion)


def cargar_todo(seed: int = SEED):
    """Pipeline completo de datos: devuelve dict con train, val, test, externo, d (limpio) y sin_contenido."""
    data = preparar(cargar_dataset())
    d, sin_contenido = limpiar(data)
    d["split"] = particion_por_plantilla(d, seed)
    out = {s: d[d.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    out.update(d=d, sin_contenido=sin_contenido,
               externo=preparar(pd.read_csv(TEST_EXTERNO, keep_default_na=False)))
    return out
