import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import (cluster_bootstrap_ci, compute_metrics,
                            critical_class_analysis, top_confusions)
from src.experiment_tracking import FIELDS, log_experiment

LABELS = ["a", "b", "c"]
Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c"]


def test_hand_computed_metrics():
    m = compute_metrics(Y_TRUE, Y_PRED, LABELS)
    # a: P=1, R=2/3, F1=0.8 ; b: P=1/2, R=1/2, F1=0.5 ; c: P=1/2, R=1, F1=2/3
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision_macro"] == pytest.approx((1 + .5 + .5) / 3)
    assert m["recall_macro"] == pytest.approx((2 / 3 + .5 + 1) / 3)
    assert m["f1_macro"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert m["f1_weighted"] == pytest.approx((0.8 * 3 + 0.5 * 2 + 2 / 3 * 1) / 6)
    assert m["per_class"]["a"]["support"] == 3


def test_confusion_matrix_and_distribution():
    m = compute_metrics(Y_TRUE, Y_PRED, LABELS)
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert m["prediction_distribution"] == {"a": 2, "b": 2, "c": 2}
    assert top_confusions(m["confusion_matrix"], LABELS, 1)[0][2] == 1


def test_class_absent_from_predictions_counts_zero_not_nan():
    m = compute_metrics(["a", "b"], ["a", "a"], LABELS)
    assert m["per_class"]["b"]["f1"] == 0.0 and not np.isnan(m["f1_macro"])


def test_critical_class_analysis():
    m = compute_metrics(Y_TRUE, Y_PRED, LABELS)
    a = critical_class_analysis(m, LABELS, critical=["b"])
    assert a["b"]["reales_clasificados_como"] == {"c": 1}
    assert a["b"]["falsas_alarmas_desde"] == {"a": 1}
    assert a["b"]["n_predichos"] == 2


def test_bootstrap_is_deterministic_and_brackets_estimate():
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]
    r1 = cluster_bootstrap_ci(Y_TRUE, Y_PRED, groups, n_boot=200, seed=1)
    r2 = cluster_bootstrap_ci(Y_TRUE, Y_PRED, groups, n_boot=200, seed=1)
    assert r1 == r2
    assert r1["f1_macro_ci_low"] <= r1["f1_macro_ci_high"]
    assert r1["n_clusters"] == 3


def test_experiment_log_writes_real_row_and_rejects_unknown(tmp_path):
    p = tmp_path / "log.csv"
    log_experiment(p, experiment_id="e1", model="m", seed=42, params={"C": 1})
    log_experiment(p, experiment_id="e2", model="m")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == FIELDS and len(lines) == 3
    with pytest.raises(ValueError):
        log_experiment(p, inventado="x")


def test_test_set_is_evaluated_only_once(tmp_path):
    from src.experiment_tracking import TestAlreadyEvaluated, guard_test_evaluation
    guard_test_evaluation(tmp_path, "m", "a" * 64, "e1")
    with pytest.raises(TestAlreadyEvaluated):
        guard_test_evaluation(tmp_path, "m", "a" * 64, "e2")
    # otra versión de datos o forzar con rastro sí se permite
    guard_test_evaluation(tmp_path, "m", "b" * 64, "e3")
    lock = guard_test_evaluation(tmp_path, "m", "a" * 64, "e4", force=True)
    import json
    assert [e["forzada"] for e in json.loads(lock.read_text())["evaluaciones"]] == [False, True]
