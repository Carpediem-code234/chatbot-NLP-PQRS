#!/usr/bin/env python3
"""Ajusta, compara y evalúa los clasificadores de PQRS."""

import json
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqrs_nlp.data import (
    EXPECTED_INTENTS,
    grouped_cv_splits,
    split_by_template,
    validate_dataset,
)
from pqrs_nlp.models import build_models, parameter_grids

SEED = 42
DATA_PATH = ROOT / "data" / "processed" / "pqrs_sinteticas_limpias.csv"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"


def _metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=EXPECTED_INTENTS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=EXPECTED_INTENTS,
                average="macro", zero_division=0
            )[0]
        ),
        "recall_macro": float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=EXPECTED_INTENTS,
                average="macro", zero_division=0
            )[1]
        ),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=EXPECTED_INTENTS,
            average="macro", zero_division=0
        )),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred, labels=EXPECTED_INTENTS,
                average="weighted", zero_division=0
            )
        ),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for label, p, r, f, n in zip(
                EXPECTED_INTENTS, precision, recall, f1, support
            )
        },
    }


def _write_json(path, content):
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    for directory in (METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} no existe. Ejecuta primero scripts/prepare_dataset.py."
        )
    frame = pd.read_csv(DATA_PATH)
    validate_dataset(frame)
    splits, split_metadata = split_by_template(frame, seed=SEED)
    train_idx = splits["train"]
    validation_idx = splits["validation"]
    test_idx = splits["test"]
    X_train = frame.loc[train_idx, "texto_usuario"]
    y_train = frame.loc[train_idx, "intencion"]
    groups_train = frame.loc[train_idx, "template_id"]
    X_validation = frame.loc[validation_idx, "texto_usuario"]
    y_validation = frame.loc[validation_idx, "intencion"]
    X_test = frame.loc[test_idx, "texto_usuario"]
    y_test = frame.loc[test_idx, "intencion"]

    cv_splits = grouped_cv_splits(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        groups_train.reset_index(drop=True),
        seed=SEED,
        n_splits=5,
    )
    models = build_models(seed=SEED)
    search_objects = {}
    tuning_rows = []
    fit_times = {}
    grids = parameter_grids()

    for name in ("logistic_regression", "word_char_linear_svc"):
        start = time.perf_counter()
        search = GridSearchCV(
            models[name],
            grids[name],
            scoring="f1_macro",
            cv=cv_splits,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fit_times[name] = time.perf_counter() - start
        search_objects[name] = search
        for index, parameters in enumerate(search.cv_results_["params"]):
            tuning_rows.append({
                "model": name,
                "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                "mean_validation_f1_macro": float(
                    search.cv_results_["mean_test_score"][index]
                ),
                "std_validation_f1_macro": float(
                    search.cv_results_["std_test_score"][index]
                ),
                "mean_fit_time_seconds": float(
                    search.cv_results_["mean_fit_time"][index]
                ),
                "is_best": index == search.best_index_,
            })

    pd.DataFrame(tuning_rows).to_csv(
        METRICS_DIR / "tuning_results.csv", index=False
    )
    _write_json(METRICS_DIR / "split_metadata.json", split_metadata)

    validation_models = {"dummy": models["dummy"]}
    validation_models.update({
        name: search.best_estimator_ for name, search in search_objects.items()
    })
    validation_predictions = {}
    validation_rows = []
    for name, model in validation_models.items():
        start = time.perf_counter()
        if name == "dummy":
            model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        validation_predictions[name] = prediction
        result = _metrics(y_validation, prediction)
        validation_rows.append({"model": name, **{
            key: value for key, value in result.items() if key != "per_class"
        }})
        _write_json(
            METRICS_DIR / f"validation_{name}_per_class.json",
            result["per_class"],
        )
        fit_times.setdefault(name, time.perf_counter() - start)
    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)
    selected_name = max(
        validation_rows, key=lambda row: (row["f1_macro"], row["model"])
    )["model"]

    selected = validation_models[selected_name]
    train_validation_idx = list(train_idx) + list(validation_idx)
    X_train_validation = frame.loc[train_validation_idx, "texto_usuario"]
    y_train_validation = frame.loc[train_validation_idx, "intencion"]
    selected.fit(X_train_validation, y_train_validation)
    test_predictions = selected.predict(X_test)
    test_result = _metrics(y_test, test_predictions)
    test_result.update({
        "selected_model": selected_name,
        "selection_metric": "F1 macro en validación",
        "partition": split_metadata,
        "training_seconds": float(fit_times.get(selected_name, 0.0)),
        "test_classification_report": classification_report(
            y_test, test_predictions, labels=EXPECTED_INTENTS,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_predictions, labels=EXPECTED_INTENTS
        ).tolist(),
    })
    _write_json(METRICS_DIR / "test_metrics.json", test_result)
    pd.DataFrame(test_result["per_class"]).T.rename_axis("intencion").to_csv(
        METRICS_DIR / "test_per_class.csv"
    )
    errors = frame.loc[test_idx, [
        "id_registro", "texto_usuario", "intencion", "nivel_lenguaje"
    ]].copy()
    errors["prediccion"] = test_predictions
    errors = errors.loc[errors["intencion"] != errors["prediccion"]]
    errors.to_csv(METRICS_DIR / "test_errors.csv", index=False)

    language_rows = []
    language_class_rows = []
    language_test = frame.loc[test_idx, "nivel_lenguaje"]
    for level in sorted(language_test.unique()):
        mask = language_test.eq(level).to_numpy()
        count = int(mask.sum())
        result = _metrics(y_test.to_numpy()[mask], test_predictions[mask])
        language_class_rows.extend(
            {
                "nivel_lenguaje": level,
                "intencion": label,
                **class_metrics,
            }
            for label, class_metrics in result["per_class"].items()
        )
        low_support = [
            label for label, class_metrics in result["per_class"].items()
            if class_metrics["support"] < 5
        ]
        language_rows.append({
            "nivel_lenguaje": level,
            "examples": count,
            **{key: result[key] for key in (
                "accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted",
            )},
            "interpretation_warning": (
                "Pocos ejemplos por nivel o clase; métricas preliminares. "
                f"Soporte menor de 5 en: {', '.join(low_support)}."
                if count < 10 or low_support else ""
            ),
        })
    language_table = pd.DataFrame(language_rows)
    language_table.to_csv(METRICS_DIR / "test_by_language.csv", index=False)
    pd.DataFrame(language_class_rows).to_csv(
        METRICS_DIR / "test_by_language_per_class.csv", index=False
    )

    sns.set_theme(style="whitegrid")
    class_counts = frame["intencion"].value_counts().reindex(EXPECTED_INTENTS)
    ax = class_counts.plot(kind="bar", color="#4472C4")
    ax.set(title="Distribución de intenciones", xlabel="Intención", ylabel="Registros")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()

    ax = validation_table.set_index("model")[[
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"
    ]].plot(kind="bar", ylim=(0, 1), figsize=(10, 5))
    ax.set(title="Comparación en validación por plantillas", ylabel="Puntuación")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_model_comparison.png", dpi=160)
    plt.close()

    matrix = confusion_matrix(y_test, test_predictions, labels=EXPECTED_INTENTS)
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=EXPECTED_INTENTS, yticklabels=EXPECTED_INTENTS,
    )
    plt.title(f"Matriz de confusión de prueba: {selected_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Intención real")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_confusion_matrix.png", dpi=160)
    plt.close()

    pd.DataFrame(test_result["per_class"]).T[["precision", "recall", "f1"]].plot(
        kind="bar", ylim=(0, 1), figsize=(9, 5)
    )
    plt.title("Rendimiento por intención en prueba")
    plt.ylabel("Puntuación")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_per_class.png", dpi=160)
    plt.close()

    language_table.set_index("nivel_lenguaje")[
        ["accuracy", "f1_macro", "f1_weighted"]
    ].plot(kind="bar", ylim=(0, 1), figsize=(8, 5))
    plt.title("Rendimiento por nivel de lenguaje en prueba")
    plt.ylabel("Puntuación")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_by_language.png", dpi=160)
    plt.close()

    metadata = {
        "artifact_warning": (
            "Artefacto experimental entrenado con datos sintéticos; "
            "no apto para atención clínica ni producción."
        ),
        "selected_model": selected_name,
        "seed": SEED,
        "training_rows": len(train_validation_idx),
        "training_template_groups": sorted(
            set(frame.loc[train_validation_idx, "template_id"])
        ),
    }
    joblib.dump(
        {"pipeline": selected, "metadata": metadata},
        MODELS_DIR / "pqrs_experimental.joblib",
    )
    _write_json(MODELS_DIR / "pqrs_experimental_metadata.json", metadata)
    print(f"Modelo seleccionado en validación: {selected_name}")
    print("Métricas finales (prueba reservada):")
    print(json.dumps({
        key: test_result[key]
        for key in ("accuracy", "precision_macro", "recall_macro",
                    "f1_macro", "f1_weighted")
    }, ensure_ascii=False, indent=2))
    print(f"Errores en prueba: {len(errors)}")
    print(f"Artefacto experimental: {MODELS_DIR / 'pqrs_experimental.joblib'}")


if __name__ == "__main__":
    main()
