"""Métricas, evaluación en varias particiones y selección del umbral de confianza."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from src.config import CLASES
from src.datos import particion_por_plantilla


def evaluar(modelo, D: pd.DataFrame, y_col: str = "tipo_pqrs") -> dict:
    y, p = D[y_col], modelo.predict(D)
    fila = {"F1_macro": f1_score(y, p, average="macro"), "accuracy": accuracy_score(y, p),
            "acc_balanceada": balanced_accuracy_score(y, p)}
    try:
        fila["AUC_macro_ovr"] = roc_auc_score(y, modelo.predict_proba(D), multi_class="ovr", average="macro",
                                              labels=modelo.classes_)
    except Exception:
        fila["AUC_macro_ovr"] = np.nan
    if y_col == "tipo_pqrs":
        pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=CLASES, zero_division=0)
        for c, a, b in zip(CLASES, pr, rc):
            fila[f"P_{c}"], fila[f"R_{c}"] = a, b
    return fila


def robustez(constructores: dict, d: pd.DataFrame, externo: pd.DataFrame, semillas=range(10)) -> pd.DataFrame:
    """Repite el experimento con varias particiones por plantilla (1 plantilla por intención a test)."""
    filas = []
    for seed in semillas:
        split = particion_por_plantilla(d, seed, con_val=False)
        tv, te = d[split == "train"], d[split == "test"]
        for nombre, ctor in constructores.items():
            m = ctor().fit(tv, tv.tipo_pqrs)
            filas.append({"semilla": seed, "modelo": nombre,
                          "test (plantillas no vistas)": f1_score(te.tipo_pqrs, m.predict(te), average="macro"),
                          "test externo": f1_score(externo.tipo_pqrs, m.predict(externo), average="macro")})
    return pd.DataFrame(filas)


def curva_umbral(modelo, D: pd.DataFrame, taus=np.round(np.arange(0.2, 0.96, 0.05), 2)) -> pd.DataFrame:
    """Cobertura (mensajes que el modelo decide solo) y accuracy de esas decisiones para cada umbral τ."""
    pr = modelo.predict_proba(D)
    conf, pred = pr.max(1), modelo.classes_[pr.argmax(1)]
    ok = pred == D.tipo_pqrs.values
    return pd.DataFrame([{"tau": t, "cobertura": (conf >= t).mean(),
                          "accuracy_decididos": ok[conf >= t].mean() if (conf >= t).any() else np.nan}
                         for t in taus])


def elegir_tau(curva: pd.DataFrame, objetivo: float = 0.90, defecto: float = 0.6) -> float:
    cand = curva[curva.accuracy_decididos >= objetivo]
    return float(cand.tau.iloc[0]) if len(cand) else defecto
