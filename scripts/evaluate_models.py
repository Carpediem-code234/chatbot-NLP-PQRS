"""Evaluación FINAL en test de modelos ya entrenados y seleccionados con dev.

Uso:  python scripts/evaluate_models.py --models baseline
Cada (modelo, versión de datos) se evalúa en test UNA sola vez (candado).
Esta etapa solo implementa el baseline; el Transformer se agregará en su etapa.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.constants import INTENTS
from src.data_validation import load_dataset
from src.evaluation import (cluster_bootstrap_ci, compute_metrics,
                            critical_class_analysis, errors_table,
                            plot_confusion_matrix, top_confusions)
from src.experiment_tracking import (file_sha256, guard_test_evaluation,
                                     hardware_info, log_experiment, new_experiment_id)
from src.preprocessing import get_xy

REGISTRY = {"baseline": ROOT / "models/baseline_tfidf_lr.joblib"}


def evaluate_baseline(art, df, out: Path):
    Xte, yte, pte = get_xy(df, "test")
    model = art["pipeline"]
    proba = model.predict_proba(Xte); cls = list(model.classes_)
    pred = [cls[i] for i in proba.argmax(1)]
    m = compute_metrics(yte, pred, INTENTS)
    m["bootstrap"] = cluster_bootstrap_ci(yte, pred, pte["template_id"].values, seed=art["seed"])
    m["critical"] = critical_class_analysis(m, INTENTS)
    m["top_confusions"] = top_confusions(m["confusion_matrix"], INTENTS, 10)
    out.mkdir(parents=True, exist_ok=True)
    errors_table(pte, pred, proba.max(1)).to_csv(out / "test_errors.csv", index=False, encoding="utf-8")
    plot_confusion_matrix(m["confusion_matrix"], INTENTS, out / "test_confusion_matrix.png",
                          "Baseline TF-IDF + LR — test (evaluación única)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline"], choices=list(REGISTRY))
    ap.add_argument("--data", default=str(ROOT / "data/raw/pqrs_sintetico.csv"))
    ap.add_argument("--force", action="store_true", help="reevaluar (queda registrado en el candado)")
    a = ap.parse_args()
    df = load_dataset(a.data); sha = file_sha256(a.data)
    review = ",".join(sorted(set(df["review_status"])))
    for name in a.models:
        art = joblib.load(REGISTRY[name])
        if art["data_sha256"] != sha:
            sys.exit(f"El modelo {name} se entrenó con otra versión de datos; reentrénelo antes de evaluar.")
        exp_id = new_experiment_id(f"eval_test_{name}")
        guard_test_evaluation(ROOT / "results", name, sha, exp_id, force=a.force)
        out = ROOT / "results" / name
        m = evaluate_baseline(art, df, out)
        (out / "test_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        log_experiment(
            ROOT / "results/experiment_log.csv", experiment_id=exp_id, model=f"{name}_tfidf_lr",
            checkpoint_id=Path(REGISTRY[name]).name, seed=art["seed"], params=art["config"],
            splits_used=f"test (evaluación final única); modelo ajustado con {art['fit_on']}",
            data_review_status=review,
            test_metrics={k: round(m[k], 4) for k in ("accuracy", "precision_macro", "recall_macro",
                                                      "f1_macro", "f1_weighted")},
            artifacts_path=f"results/{name}/test_metrics.json; test_errors.csv; test_confusion_matrix.png",
            limitations="Preliminar: datos sintéticos pendientes de revisión humana; test con 48 plantillas "
                        "(n efectivo bajo); no representa usuarios reales.",
            hardware=hardware_info())
        print(f"{name}: TEST F1 macro={m['f1_macro']:.4f} acc={m['accuracy']:.4f} "
              f"IC95% plantillas=[{m['bootstrap']['f1_macro_ci_low']:.3f}, {m['bootstrap']['f1_macro_ci_high']:.3f}]")


if __name__ == "__main__":
    main()
