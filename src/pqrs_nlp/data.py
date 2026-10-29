"""Lectura, validación, limpieza y partición grupal del corpus."""

import math

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .preprocessing import normalize_text

EXPECTED_INTENTS = (
    "solicitar_cita",
    "queja_atencion",
    "reclamo_medicamentos",
    "consultar_autorizacion",
)
REQUIRED_COLUMNS = ("texto_usuario", "intencion", "nivel_lenguaje", "entidades_etiquetadas")


def validate_dataset(frame, require_all_classes=True):
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    if frame["intencion"].isna().any():
        raise ValueError("Hay etiquetas intencion nulas.")
    labels = set(frame["intencion"].astype(str))
    unexpected = labels - set(EXPECTED_INTENTS)
    if unexpected:
        raise ValueError(f"Etiquetas no reconocidas: {sorted(unexpected)}")
    missing_labels = set(EXPECTED_INTENTS) - labels
    if require_all_classes and missing_labels:
        raise ValueError(f"Faltan intenciones esperadas: {sorted(missing_labels)}")
    return {
        "rows": int(len(frame)),
        "labels": sorted(labels),
        "missing_expected_labels": sorted(missing_labels),
        "null_texts": int(frame["texto_usuario"].isna().sum()),
        "blank_texts": int(
            frame["texto_usuario"].fillna("").astype(str).str.strip().eq("").sum()
        ),
    }


def prepare_dataset(frame):
    """Descarta textos inválidos y duplicados, detectando conflictos de etiqueta."""
    validate_dataset(frame, require_all_classes=False)
    result = frame.copy()
    result["texto_usuario"] = result["texto_usuario"].fillna("").astype(str)
    result["_texto_clave"] = result["texto_usuario"].map(normalize_text)
    invalid = result["_texto_clave"].eq("")
    valid = result.loc[~invalid].copy()

    label_counts = valid.groupby("_texto_clave")["intencion"].nunique()
    conflicts = label_counts[label_counts > 1]
    if not conflicts.empty:
        examples = valid.loc[
            valid["_texto_clave"].isin(conflicts.index),
            ["texto_usuario", "intencion"],
        ].head(5)
        raise ValueError(
            "Textos idénticos con etiquetas incompatibles: "
            f"{examples.to_dict(orient='records')}"
        )

    duplicates = int(valid.duplicated("_texto_clave", keep="first").sum())
    valid = valid.drop_duplicates("_texto_clave", keep="first").copy()
    valid["texto_normalizado"] = valid["_texto_clave"]
    valid = valid.drop(columns="_texto_clave").reset_index(drop=True)
    if "template_id" not in valid.columns:
        valid["template_id"] = pd.NA
    summary = {
        "input_rows": int(len(frame)),
        "null_text_rows_discarded": int(frame["texto_usuario"].isna().sum()),
        "blank_text_rows_discarded": int(
            frame["texto_usuario"].fillna("").astype(str).str.strip().eq("").sum()
            - frame["texto_usuario"].isna().sum()
        ),
        "duplicate_rows_discarded": duplicates,
        "output_rows": int(len(valid)),
        "conflicting_text_label_groups": 0,
    }
    return valid, summary


def split_by_template(frame, seed=42, fractions=(0.65, 0.15, 0.20)):
    """Asigna plantillas completas a train/validation/test dentro de cada clase."""
    if "template_id" not in frame or frame["template_id"].isna().any():
        raise ValueError(
            "No se puede evaluar fuga por plantillas: template_id falta o contiene nulos."
        )
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
        raise ValueError("Se requieren tres fracciones que sumen 1.")

    group_labels = frame.groupby("template_id", sort=True)["intencion"].nunique()
    if (group_labels != 1).any():
        raise ValueError("Cada template_id debe corresponder a una sola intencion.")
    groups_by_label = (
        frame.drop_duplicates("template_id")
        .groupby("intencion")["template_id"]
        .apply(list)
    )
    group_sizes = frame.groupby("template_id").size().to_dict()
    groups_by_label = {
        label: list(groups_by_label[label])
        for label in sorted(frame["intencion"].unique())
    }
    for label, groups in groups_by_label.items():
        if len(groups) < 5:
            raise ValueError(
                f"La clase {label} solo tiene {len(groups)} plantillas; "
                "se requieren al menos 5 para train/validación/test y CV."
            )

    import random

    rng = random.Random(seed)
    names = ("train", "validation", "test")
    total_rows = len(frame)
    target_rows = [fraction * total_rows for fraction in fractions]
    best_assignment = None
    best_error = float("inf")
    for _ in range(10_000):
        candidate = {name: set() for name in names}
        row_counts = [0, 0, 0]
        for groups in groups_by_label.values():
            shuffled = groups.copy()
            rng.shuffle(shuffled)
            per_split = [[], [], []]
            for group_id in shuffled:
                split_index = rng.choices(range(3), weights=fractions, k=1)[0]
                per_split[split_index].append(group_id)
            if any(not assigned_groups for assigned_groups in per_split):
                break
            for split_index, assigned_groups in enumerate(per_split):
                candidate[names[split_index]].update(assigned_groups)
                row_counts[split_index] += sum(
                    group_sizes[group_id] for group_id in assigned_groups
                )
        else:
            error = sum(
                ((count - target) / total_rows) ** 2
                for count, target in zip(row_counts, target_rows)
            )
            if error < best_error:
                best_error = error
                best_assignment = candidate

    if best_assignment is None:
        raise ValueError(
            "No se encontró una asignación con todas las clases en cada partición."
        )
    assigned = best_assignment
    actual_fractions = {
        name: sum(group_sizes[group_id] for group_id in group_ids) / total_rows
        for name, group_ids in assigned.items()
    }
    if any(
        abs(actual_fractions[name] - fractions[index]) > 0.02
        for index, name in enumerate(names)
    ):
        raise ValueError(
            "La separación por grupos no permite aproximarse a las proporciones "
            f"solicitadas; proporciones alcanzadas: {actual_fractions}."
        )

    splits = {
        name: frame.index[frame["template_id"].isin(group_ids)].to_numpy()
        for name, group_ids in assigned.items()
    }
    if any(len(splits[name]) == 0 for name in splits):
        raise ValueError("La partición dejó un conjunto vacío.")
    for name, indices in splits.items():
        labels = set(frame.loc[indices, "intencion"])
        if labels != set(frame["intencion"]):
            raise ValueError(f"Faltan clases en la partición {name}: {labels}")
    overlap = {
        f"{left}_{right}": sorted(assigned[left] & assigned[right])
        for left, right in (("train", "validation"), ("train", "test"),
                            ("validation", "test"))
    }
    if any(overlap.values()):
        raise AssertionError(f"Hay grupos de plantilla compartidos: {overlap}")
    return splits, {
        "seed": seed,
        "method": "partición estratificada por clase y template_id",
        "fractions_requested": {
            "train": fractions[0],
            "validation": fractions[1],
            "test": fractions[2],
        },
        "rows": {name: int(len(indices)) for name, indices in splits.items()},
        "fractions_actual": {
            name: float(actual_fractions[name]) for name in names
        },
        "groups": {name: int(len(group_ids)) for name, group_ids in assigned.items()},
        "shared_groups": overlap,
    }


def grouped_cv_splits(X, y, groups, seed=42, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups))
    labels = set(y)
    for fold, (train_idx, validation_idx) in enumerate(splits, start=1):
        if set(y.iloc[train_idx]) != labels or set(y.iloc[validation_idx]) != labels:
            raise ValueError(
                f"El pliegue {fold} no contiene las cuatro clases en ambos conjuntos."
            )
        if set(groups.iloc[train_idx]) & set(groups.iloc[validation_idx]):
            raise ValueError(f"Hay fuga de plantilla en el pliegue {fold}.")
    return splits
