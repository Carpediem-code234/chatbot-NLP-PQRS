"""Registro de experimentos REALES en results/experiment_log.csv (spec §9)."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "experiment_id", "timestamp_utc", "model", "checkpoint_id", "seed", "params",
    "splits_used", "data_review_status", "val_metrics", "test_metrics",
    "artifacts_path", "limitations", "hardware",
]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware_info() -> dict:
    import os
    import sklearn
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "sklearn": sklearn.__version__}
    try:  # torch es opcional (solo Transformer)
        import torch
        info.update({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available())})
    except Exception:
        info["torch"] = "no instalado"
    return info


def new_experiment_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def log_experiment(log_path, **row) -> dict:
    """Agrega una fila. Los campos dict/list se serializan a JSON. Falta -> ''."""
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError(f"Campos desconocidos: {sorted(unknown)}")
    row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row = {k: (json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v)
           for k, v in row.items()}
    path = Path(log_path); path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return row


class TestAlreadyEvaluated(RuntimeError):
    pass


def guard_test_evaluation(lock_dir, model_name: str, data_sha256: str, experiment_id: str,
                          force: bool = False) -> Path:
    """Impide evaluar el test más de una vez por (modelo, versión de datos).
    Con force=True se permite pero queda registrado en el propio lock (auditable)."""
    lock_dir = Path(lock_dir); lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f".test_eval_{model_name}_{data_sha256[:12]}.json"
    if lock.exists() and not force:
        raise TestAlreadyEvaluated(
            f"El test ya se evaluó para {model_name} con estos datos ({lock.name}). "
            "Reevaluar para 'mejorar' métricas invalida la comparación; use force solo con justificación.")
    prev = json.loads(lock.read_text()) if lock.exists() else {"evaluaciones": []}
    prev["evaluaciones"].append({"experiment_id": experiment_id,
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "forzada": bool(force)})
    lock.write_text(json.dumps(prev, indent=2))
    return lock
