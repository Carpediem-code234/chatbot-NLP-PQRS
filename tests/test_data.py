import pandas as pd
import pytest

from pqrs_nlp.data import (
    EXPECTED_INTENTS,
    prepare_dataset,
    split_by_template,
    validate_dataset,
)
from pqrs_nlp.synthetic import generate_synthetic_dataframe


def test_synthetic_generation_and_cleaning_are_reproducible():
    raw = generate_synthetic_dataframe(seed=42)
    clean, summary = prepare_dataset(raw)
    clean_again, _ = prepare_dataset(generate_synthetic_dataframe(seed=42))
    assert len(raw) > len(clean)
    assert summary["duplicate_rows_discarded"] > 0
    assert summary["null_text_rows_discarded"] > 0
    assert clean["texto_usuario"].str.strip().ne("").all()
    assert clean["intencion"].isin(EXPECTED_INTENTS).all()
    assert clean["texto_usuario"].tolist() == clean_again["texto_usuario"].tolist()
    assert {"id_registro", "template_id", "texto_original", "nivel_lenguaje"} <= set(clean)
    validate_dataset(clean)


def test_conflicting_labels_for_same_normalized_text_fail():
    frame = pd.DataFrame([
        {
            "texto_usuario": "No me han dado medicamento",
            "intencion": "reclamo_medicamentos",
            "nivel_lenguaje": "simplificado",
            "entidades_etiquetadas": "",
        },
        {
            "texto_usuario": "  no me han dado medicamento ",
            "intencion": "queja_atencion",
            "nivel_lenguaje": "simplificado",
            "entidades_etiquetadas": "",
        },
    ])
    with pytest.raises(ValueError, match="etiquetas incompatibles"):
        prepare_dataset(frame)


def test_template_groups_are_disjoint_and_all_classes_are_present():
    clean, _ = prepare_dataset(generate_synthetic_dataframe())
    splits, metadata = split_by_template(clean)
    group_sets = {
        name: set(clean.loc[indices, "template_id"])
        for name, indices in splits.items()
    }
    assert not group_sets["train"] & group_sets["validation"]
    assert not group_sets["train"] & group_sets["test"]
    assert not group_sets["validation"] & group_sets["test"]
    for indices in splits.values():
        assert set(clean.loc[indices, "intencion"]) == set(EXPECTED_INTENTS)
    assert 0.63 <= metadata["fractions_actual"]["train"] <= 0.67
    assert 0.13 <= metadata["fractions_actual"]["validation"] <= 0.17
    assert 0.18 <= metadata["fractions_actual"]["test"] <= 0.22
    assert metadata["shared_groups"] == {
        "train_validation": [],
        "train_test": [],
        "validation_test": [],
    }


def test_missing_template_ids_cannot_be_used_for_independent_split():
    frame = pd.DataFrame([
        {
            "texto_usuario": "mensaje",
            "intencion": EXPECTED_INTENTS[0],
            "nivel_lenguaje": "formal",
            "entidades_etiquetadas": "",
            "template_id": None,
        },
    ])
    with pytest.raises(ValueError, match="template_id"):
        split_by_template(frame)
