"""Métricas, análisis de clases críticas, errores y figuras (spec §10, §12)."""
from __future__ import annotations

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .constants import CRITICAL_INTENTS


def compute_metrics(y_true, y_pred, labels) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    n = len(y_true)
    return {
        "n": n,
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "f1_weighted": float(np.average(f, weights=s)) if s.sum() else 0.0,
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "prediction_distribution": dict(Counter(y_pred)),
    }


def top_confusions(cm, labels, k=10):
    cm = np.array(cm)
    pairs = [(labels[i], labels[j], int(cm[i, j]))
             for i in range(len(labels)) for j in range(len(labels))
             if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:k]


def critical_class_analysis(metrics: dict, labels, critical=CRITICAL_INTENTS) -> dict:
    cm = np.array(metrics["confusion_matrix"])
    out = {}
    for c in critical:
        i = labels.index(c)
        missed = {labels[j]: int(cm[i, j]) for j in range(len(labels)) if j != i and cm[i, j]}
        false_alarms = {labels[j]: int(cm[j, i]) for j in range(len(labels)) if j != i and cm[j, i]}
        out[c] = {**metrics["per_class"][c],
                  "n_predichos": int(cm[:, i].sum()),
                  "reales_clasificados_como": missed,
                  "falsas_alarmas_desde": false_alarms}
    return out


def cluster_bootstrap_ci(y_true, y_pred, groups, n_boot=1000, seed=42, alpha=0.05):
    """IC percentil del F1 macro remuestreando PLANTILLAS (no filas): las filas de
    una misma plantilla son casi idénticas y no son observaciones independientes."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        _, _, f, _ = precision_recall_fscore_support(
            y_true[ii], y_pred[ii], average="macro", zero_division=0)
        vals.append(f)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"f1_macro_ci_low": float(lo), "f1_macro_ci_high": float(hi),
            "n_boot": n_boot, "n_clusters": int(len(uniq)), "seed": seed}


def errors_table(part: pd.DataFrame, y_pred, confidence=None) -> pd.DataFrame:
    t = part[["id", "text", "intent", "style", "template_id"]].copy()
    t = t.rename(columns={"intent": "etiqueta_real"})
    t["prediccion"] = list(y_pred)
    if confidence is not None:
        t["confianza"] = np.round(confidence, 4)
    return t[t["etiqueta_real"] != t["prediccion"]].reset_index(drop=True)


def plot_confusion_matrix(cm, labels, path, title, normalize=True):
    cm = np.array(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if shown[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
