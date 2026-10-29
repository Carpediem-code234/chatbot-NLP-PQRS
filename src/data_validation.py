"""Validación del dataset PQRS sintético (Fase 2, etapa 1).

Cada comprobación devuelve datos medidos; nada se asume. La normalización
usada aquí (minúsculas, sin tildes ni puntuación) sirve SOLO para detectar
duplicados: el texto original nunca se modifica y el preprocesamiento del
modelo conserva tildes y negaciones.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import (
    CRITICAL_INTENTS, INTENTS, REQUIRED_COLUMNS, VALID_SPLITS, VALID_STYLES,
)

# safety_flags esperado por intención según lo observado en la Fase 1.
EXPECTED_FLAGS = {
    "urgencia_medica": "urgencia_medica;prioridad_alta",
    "hablar_agente": "escalamiento_humano",
    "no_entendido": "baja_confianza",
}

# Léxicos heurísticos solo para señalar posibles conflictos texto-etiqueta
# (no son verdad de terreno; cada hallazgo requiere revisión humana).
URGENT_PATTERNS = [
    r"ambulancia", r"no (puedo )?respir", r"desmay", r"emergencia", r"auxilio",
    r"duele.*pecho", r"no se mueve", r"urgente", r"sangr", r"convuls",
]
AGENT_PATTERNS = [
    r"\bagente\b", r"\basesor\b", r"\bfuncionario\b", r"\bhumano\b",
    r"\bpersona\b", r"alguien de verdad", r"\brobot\b",
]


def normalize_for_dedup(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_dataset(path: str) -> pd.DataFrame:
    if str(path).endswith(".jsonl"):
        return pd.read_json(path, lines=True, dtype=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _add(report: dict, name: str, status: str, detail: Any) -> None:
    """status: ok | warning | critical"""
    report["checks"][name] = {"status": status, "detail": detail}


def validate(df: pd.DataFrame, seed: int = 42) -> dict:
    rep: dict[str, Any] = {"checks": {}, "n_rows": int(len(df))}

    # 1. Columnas y tipos
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    _add(rep, "columnas", "critical" if missing else "ok",
         {"faltantes": missing, "extra": extra})
    if missing:
        return rep
    non_str = [c for c in REQUIRED_COLUMNS if not all(isinstance(v, str) for v in df[c])]
    _add(rep, "tipos_texto", "critical" if non_str else "ok", {"columnas_no_texto": non_str})

    d = df.copy()
    d["_norm"] = d["text"].map(normalize_for_dedup)

    # 2. Vacíos, ids, etiquetas, splits, estilos
    empty = d[(d["text"].str.strip() == "") | (d["_norm"] == "")]
    _add(rep, "textos_vacios", "critical" if len(empty) else "ok", {"n": int(len(empty))})
    _add(rep, "ids_unicos", "ok" if d["id"].is_unique else "critical",
         {"duplicados": int(d["id"].duplicated().sum())})
    bad_int = sorted(set(d["intent"]) - set(INTENTS))
    absent = sorted(set(INTENTS) - set(d["intent"]))
    _add(rep, "etiquetas_validas", "critical" if (bad_int or absent) else "ok",
         {"invalidas": bad_int, "ausentes": absent, "n_clases": int(d["intent"].nunique())})
    bad_split = sorted(set(d["split"]) - set(VALID_SPLITS))
    _add(rep, "splits_validos", "critical" if bad_split else "ok",
         {"invalidos": bad_split, "conteo": d["split"].value_counts().to_dict()})
    bad_style = sorted(set(d["style"]) - set(VALID_STYLES))
    _add(rep, "estilos_validos", "warning" if bad_style else "ok", {"invalidos": bad_style})

    # 3. Distribución y cobertura por intención y partición
    ct = pd.crosstab(d["intent"], d["split"]).reindex(columns=VALID_SPLITS, fill_value=0)
    rep["distribucion_intencion_split"] = ct.to_dict(orient="index")
    zero = [(i, s) for i in ct.index for s in VALID_SPLITS if ct.loc[i, s] == 0]
    low = [(i, s, int(ct.loc[i, s])) for i in ct.index for s in ("dev", "test")
           if 0 < ct.loc[i, s] < 5]
    _add(rep, "cobertura_intencion_split", "critical" if zero else ("warning" if low else "ok"),
         {"sin_ejemplos": zero, "menos_de_5": low,
          "min_por_celda": int(ct.values.min())})

    # 4. Duplicados
    exact = d[d["text"].duplicated(keep=False)]
    _add(rep, "duplicados_exactos", "warning" if len(exact) else "ok",
         {"filas": int(len(exact)), "ids": exact["id"].tolist()})
    g = d.groupby("_norm").agg(n=("id", "size"), n_split=("split", "nunique"),
                               n_int=("intent", "nunique"))
    dup_groups = g[g["n"] > 1]
    rows_in_dup = int(dup_groups["n"].sum())
    _add(rep, "duplicados_normalizados",
         "warning" if len(dup_groups) else "ok",
         {"grupos": int(len(dup_groups)), "filas_en_grupos": rows_in_dup,
          "filas_redundantes": int(rows_in_dup - len(dup_groups)),
          "textos_unicos": int(d["_norm"].nunique()),
          "unicos_por_split": d.groupby("split")["_norm"].nunique().to_dict(),
          "grupos_que_cruzan_splits": int((g["n_split"] > 1).sum()),
          "grupos_con_etiquetas_distintas": int((g["n_int"] > 1).sum())})
    _add(rep, "conflicto_mismo_texto_distinta_etiqueta",
         "critical" if (g["n_int"] > 1).any() else "ok",
         {"n": int((g["n_int"] > 1).sum())})

    # 5. Fuga por plantilla y estructura del split
    ts = d.groupby("template_id")["split"].nunique()
    _add(rep, "fuga_template_id", "critical" if (ts > 1).any() else "ok",
         {"plantillas": int(d["template_id"].nunique()),
          "plantillas_en_mas_de_un_split": int((ts > 1).sum()),
          "plantillas_por_split": d.groupby("split")["template_id"].nunique().to_dict()})
    ti = d.groupby("template_id")["intent"].nunique()
    _add(rep, "plantilla_una_sola_intencion", "critical" if (ti > 1).any() else "ok",
         {"plantillas_con_varias_intenciones": int((ti > 1).sum())})
    d["_tidx"] = d["template_id"].str.extract(r"__(\d+)$")[0]
    det = d.groupby("_tidx")["split"].nunique()
    split_by_idx = (pd.crosstab(d["_tidx"], d["split"]).to_dict(orient="index"))
    _add(rep, "split_determinado_por_indice_plantilla",
         "warning" if (det == 1).all() else "ok",
         {"descripcion": "cada índice final de template_id cae en una sola partición",
          "tabla": split_by_idx,
          "filas_por_plantilla": d.groupby("template_id").size().describe().round(2).to_dict()})

    # 6. Solapamiento léxico dev/test frente a train (TF-IDF de caracteres)
    tr = d[d["split"] == "train"]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(tr["_norm"])
    Xtr = vec.transform(tr["_norm"])
    nn_stats = {}
    for sp in ("dev", "test"):
        e = d[d["split"] == sp]
        S = cosine_similarity(vec.transform(e["_norm"]), Xtr)
        mx, arg = S.max(1), S.argmax(1)
        nn_stats[sp] = {
            "sim_max_media": round(float(mx.mean()), 3),
            "sim_max_mediana": round(float(np.median(mx)), 3),
            "n_sim_ge_0.9": int((mx >= 0.9).sum()),
            "n_sim_ge_0.8": int((mx >= 0.8).sum()),
            "acierto_1nn_en_train": round(float((tr["intent"].values[arg] == e["intent"].values).mean()), 3),
            "alto_parecido_distinta_etiqueta": int(((mx >= 0.8) & (tr["intent"].values[arg] != e["intent"].values)).sum()),
        }
    _add(rep, "solapamiento_dev_test_vs_train",
         "warning" if any(v["n_sim_ge_0.9"] for v in nn_stats.values()) else "ok", nn_stats)

    # 7. Revisión humana y wayuunaiki
    rs = d["review_status"].value_counts().to_dict()
    pending = int((d["review_status"] != "revisado").sum())
    _add(rep, "review_status", "warning" if pending else "ok",
         {"conteo": rs, "filas_no_revisadas": pending,
          "nota": "Resultados deben marcarse como sintéticos pendientes de validación."
                  if pending else ""})
    _add(rep, "wayuu_status", "ok", {"conteo": d["wayuu_status"].value_counts().to_dict()})
    _add(rep, "idioma", "ok" if set(d["language"]) == {"es"} else "warning",
         {"conteo": d["language"].value_counts().to_dict()})

    # 8. safety_flags: columna derivada de la etiqueta (riesgo de fuga si se usa como feature)
    flag_map = d.groupby("intent")["safety_flags"].agg(lambda s: sorted(set(s)))
    mism = {i: flag_map[i] for i, exp in EXPECTED_FLAGS.items() if flag_map[i] != [exp]}
    other_flagged = {i: v for i, v in flag_map.items()
                     if i not in EXPECTED_FLAGS and v != ["ninguna"]}
    n_flags_per_intent = int(d.groupby("safety_flags")["intent"].nunique().max())
    _add(rep, "safety_flags_vs_intencion",
         "warning",
         {"desviaciones_esperadas": mism, "otras_intenciones_con_bandera": other_flagged,
          "mapeo": flag_map.to_dict(),
          "max_intenciones_por_valor_de_bandera": n_flags_per_intent,
          "conclusion": "safety_flags identifica 3 intenciones sin ambigüedad: NO usar como característica; "
                        "solo como referencia para la capa de reglas y auditoría."})

    # 9. Conflictos léxicos texto-etiqueta (heurísticos)
    low_txt = d["text"].str.lower()
    urg_re = re.compile("|".join(URGENT_PATTERNS))
    ag_re = re.compile("|".join(AGENT_PATTERNS))
    urg_other = d[low_txt.map(lambda t: bool(urg_re.search(t))) & (d["intent"] != "urgencia_medica")]
    urg_missing = d[(d["intent"] == "urgencia_medica") & ~low_txt.map(lambda t: bool(urg_re.search(t)))]
    ag_other = d[low_txt.map(lambda t: bool(ag_re.search(t))) & (d["intent"] != "hablar_agente")]
    ag_missing = d[(d["intent"] == "hablar_agente") & ~low_txt.map(lambda t: bool(ag_re.search(t)))]
    cols = ["id", "text", "intent", "split"]
    _add(rep, "conflictos_lexicos_heuristicos",
         "warning" if (len(urg_other) or len(urg_missing) or len(ag_other) or len(ag_missing)) else "ok",
         {"urgencia_lexico_en_otra_intencion": urg_other[cols].to_dict("records"),
          "urgencia_sin_lexico": urg_missing[cols].to_dict("records"),
          "agente_lexico_en_otra_intencion": ag_other[cols].to_dict("records"),
          "agente_sin_lexico": ag_missing[cols].to_dict("records"),
          "nota": "Heurístico: cada caso requiere revisión humana; no implica error de etiqueta."})

    # 10. Posibles artefactos del generador (puntuación final, longitud)
    d["_end"] = d["text"].str.strip().str[-1].map(
        lambda c: "!" if c == "!" else ("." if c == "." else ("?" if c == "?" else "otro")))
    end_by_int = pd.crosstab(d["intent"], d["_end"], normalize="index").round(2)
    d["_nw"] = d["text"].str.split().map(len)
    _add(rep, "artefactos_generador", "ok",
         {"cierre_por_intencion_max_exclamacion": float(end_by_int.get("!", pd.Series([0])).max()),
          "palabras_por_estilo": d.groupby("style")["_nw"].mean().round(1).to_dict(),
          "palabras_por_intencion_min_max": [int(d.groupby('intent')['_nw'].mean().round(1).min() * 10) / 10,
                                             int(d.groupby('intent')['_nw'].mean().round(1).max() * 10) / 10]})

    # Resumen
    statuses = [c["status"] for c in rep["checks"].values()]
    rep["resumen"] = {"critical": statuses.count("critical"),
                      "warning": statuses.count("warning"),
                      "ok": statuses.count("ok")}
    rep["intents_criticos"] = CRITICAL_INTENTS
    return rep
