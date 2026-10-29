"""Entrena el baseline TF-IDF + LR.  El TEST NO SE CARGA en este proceso.

1) Cada configuración de la malla se evalúa con StratifiedGroupKFold (3 pliegues,
   grupos = template_id) sobre train, y se ajusta en train completo para puntuarse en dev.
2) Selección por F1 macro en dev (spec §6); desempate: F1 macro CV, menor C.
3) Se guarda el pipeline final (ajustado SOLO con train), métricas dev y errores dev.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.baseline_model import build_pipeline, param_grid
from src.constants import INTENTS, SEED
from src.data_validation import load_dataset
from src.evaluation import (cluster_bootstrap_ci, compute_metrics,
                            critical_class_analysis, errors_table,
                            plot_confusion_matrix, top_confusions)
from src.experiment_tracking import (file_sha256, hardware_info,
                                     log_experiment, new_experiment_id)
from src.preprocessing import get_xy, select_train_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/pqrs_sintetico.csv"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(ROOT / "results/baseline"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = select_train_dev(load_dataset(a.data))        # el test se descarta aquí
    assert "test" not in set(df["split"])
    Xtr, ytr, ptr = get_xy(df, "train")
    Xdv, ydv, pdv = get_xy(df, "dev")
    groups = ptr["template_id"].values
    review = ",".join(sorted(set(df["review_status"])))
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=a.seed)
    folds = list(cv.split(Xtr, ytr, groups))

    rows, t0 = [], time.time()
    for k, cfg in enumerate(param_grid()):
        cv_f1 = []
        for tr_i, va_i in folds:
            m = build_pipeline(seed=a.seed, **cfg).fit([Xtr[i] for i in tr_i], [ytr[i] for i in tr_i])
            cv_f1.append(f1_score([ytr[i] for i in va_i], m.predict([Xtr[i] for i in va_i]),
                                  average="macro", zero_division=0))
        m = build_pipeline(seed=a.seed, **cfg).fit(Xtr, ytr)
        dv = f1_score(ydv, m.predict(Xdv), average="macro", zero_division=0)
        rows.append({**cfg, "cv_f1_macro_mean": float(np.mean(cv_f1)),
                     "cv_f1_macro_std": float(np.std(cv_f1)), "dev_f1_macro": float(dv)})
        print(f"[{k+1:02d}] {cfg} cv={np.mean(cv_f1):.3f} dev={dv:.3f}", flush=True)
    grid = pd.DataFrame(rows)
    grid["class_weight"] = grid["class_weight"].fillna("none")
    grid = grid.sort_values(["dev_f1_macro", "cv_f1_macro_mean", "C"],
                            ascending=[False, False, True]).reset_index(drop=True)
    grid.to_csv(out / "grid_results.csv", index=False)
    best = grid.iloc[0].to_dict()
    cfg = {"features": best["features"], "ngram": best["ngram"], "C": float(best["C"]),
           "class_weight": None if best["class_weight"] == "none" else best["class_weight"]}
    n_tied = int((grid["dev_f1_macro"] >= best["dev_f1_macro"] - 0.01).sum())

    model = build_pipeline(seed=a.seed, **cfg).fit(Xtr, ytr)   # solo train
    exp_id = new_experiment_id("baseline_tfidf_lr")
    model_path = ROOT / "models" / "baseline_tfidf_lr.joblib"
    joblib.dump({"pipeline": model, "config": cfg, "labels": INTENTS, "seed": a.seed,
                 "experiment_id": exp_id, "fit_on": "train",
                 "data_sha256": file_sha256(a.data)}, model_path)

    proba = model.predict_proba(Xdv); classes = list(model.classes_)
    pred = [classes[i] for i in proba.argmax(1)]
    m = compute_metrics(ydv, pred, INTENTS)
    m["bootstrap"] = cluster_bootstrap_ci(ydv, pred, pdv["template_id"].values, seed=a.seed)
    m["critical"] = critical_class_analysis(m, INTENTS)
    m["top_confusions"] = top_confusions(m["confusion_matrix"], INTENTS, 10)
    m["nota"] = ("Dev se usó para seleccionar la configuración: estas cifras son optimistas. "
                 "Datos sintéticos con review_status=" + review + ".")
    (out / "dev_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    errs = errors_table(pdv, pred, proba.max(1))
    errs.to_csv(out / "dev_errors.csv", index=False, encoding="utf-8")
    plot_confusion_matrix(m["confusion_matrix"], INTENTS, out / "dev_confusion_matrix.png",
                          "Baseline TF-IDF + LR — dev (config. seleccionada en dev)")
    (out / "selected_config.json").write_text(json.dumps(
        {"config": cfg, "dev_f1_macro": best["dev_f1_macro"], "cv_f1_macro_mean": best["cv_f1_macro_mean"],
         "configs_dentro_de_0.01_de_la_mejor_en_dev": n_tied, "seed": a.seed}, indent=2))

    log_experiment(
        ROOT / "results/experiment_log.csv", experiment_id=exp_id, model="baseline_tfidf_lr",
        checkpoint_id=model_path.name, seed=a.seed, params=cfg,
        splits_used="fit=train; seleccion=dev (+CV agrupado en train); test=NO usado",
        data_review_status=review,
        val_metrics={k: round(m[k], 4) for k in ("accuracy", "precision_macro", "recall_macro",
                                                 "f1_macro", "f1_weighted")},
        artifacts_path="models/baseline_tfidf_lr.joblib; results/baseline/",
        limitations=f"Dataset sintético pendiente de revisión humana; dev seleccionado sobre {len(grid)} configs "
                    f"({n_tied} dentro de 0.01 F1); dev tiene 48 plantillas; métricas dev optimistas.",
        hardware=hardware_info())
    print(f"\nSeleccionada: {cfg}\nDEV  F1 macro={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}  "
          f"IC95% plantillas=[{m['bootstrap']['f1_macro_ci_low']:.3f}, {m['bootstrap']['f1_macro_ci_high']:.3f}]")
    print(f"Configs dentro de 0.01 F1 de la mejor en dev: {n_tied}/{len(grid)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
