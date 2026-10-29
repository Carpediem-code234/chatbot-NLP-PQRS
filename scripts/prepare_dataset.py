#!/usr/bin/env python3
"""Genera y valida el CSV sintético de modelado de PQRS."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqrs_nlp.data import prepare_dataset, validate_dataset
from pqrs_nlp.synthetic import SEED, generate_synthetic_dataframe


def main():
    raw = generate_synthetic_dataframe(seed=SEED)
    clean, quality = prepare_dataset(raw)
    quality.update(validate_dataset(clean))
    output = ROOT / "data" / "processed" / "pqrs_sinteticas_limpias.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False, encoding="utf-8")
    quality_path = output.parent / "data_quality.json"
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset listo: {output.relative_to(ROOT)} ({len(clean)} filas)")
    print(f"Control de calidad: {quality_path.relative_to(ROOT)}")
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
