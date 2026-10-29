"""Valida el dataset y genera informe + hoja de auditoría humana.

Uso:  python scripts/validate_data.py [--data data/raw/pqrs_sintetico.csv]
Salida: results/data_validation/{report.json, report.md}
        data/processed/audit_sheet.csv  (campos de revisor VACÍOS a propósito)
Código de salida: 1 si hay comprobaciones críticas fallidas.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.constants import CRITICAL_INTENTS, SEED
from src.data_validation import load_dataset, normalize_for_dedup, validate


def build_audit_sheet(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Test completo + urgencia_medica + hablar_agente + muestra estratificada
    de train y dev (4 y 3 textos únicos por intención, semilla fija)."""
    d = df.copy()
    d["_norm"] = d["text"].map(normalize_for_dedup)
    d["grupo_duplicado"] = d.groupby("_norm").ngroup()
    parts = {"test_completo": d[d["split"] == "test"],
             "clase_critica": d[d["intent"].isin(["urgencia_medica", "hablar_agente"])
                                & d["split"].isin(["train", "dev"])]}
    for sp, k in (("train", 4), ("dev", 3)):
        e = d[(d["split"] == sp)].drop_duplicates("_norm")
        parts[f"muestra_{sp}"] = (e.groupby("intent", group_keys=False)
                                   .apply(lambda g: g.sample(min(k, len(g)), random_state=seed)))
    out = []
    for motivo, p in parts.items():
        q = p.copy(); q["motivo_auditoria"] = motivo; out.append(q)
    a = pd.concat(out)
    # una fila por id; si cae en varios motivos se concatenan
    a = (a.groupby("id", as_index=False)
          .agg({"text": "first", "intent": "first", "split": "first", "style": "first",
                "template_id": "first", "grupo_duplicado": "first",
                "motivo_auditoria": lambda s: "|".join(sorted(set(s)))}))
    a["etiqueta_revisor"] = ""      # a completar por revisor humano
    a["acuerdo_etiqueta"] = ""      # si / no / ambigua
    a["notas_revisor"] = ""
    a["revisor"] = ""
    return a.sort_values(["split", "intent", "id"]).reset_index(drop=True)


def to_markdown(rep: dict) -> str:
    L = ["# Informe de validación de datos (Fase 2, etapa 1)", "",
         f"Filas: {rep['n_rows']} | críticas: {rep['resumen']['critical']} | "
         f"advertencias: {rep['resumen']['warning']} | ok: {rep['resumen']['ok']}", "",
         "| Comprobación | Estado | Resumen |", "|---|---|---|"]
    for k, v in rep["checks"].items():
        s = json.dumps(v["detail"], ensure_ascii=False)
        L.append(f"| {k} | {v['status']} | {s[:200].replace('|', '/')} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/pqrs_sintetico.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/data_validation"))
    a = ap.parse_args()
    df = load_dataset(a.data)
    rep = validate(df)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(rep), encoding="utf-8")
    if rep["resumen"]["critical"] == 0 or "columnas" not in rep["checks"] or \
            rep["checks"]["columnas"]["status"] != "critical":
        audit = build_audit_sheet(df)
        proc = ROOT / "data/processed"; proc.mkdir(parents=True, exist_ok=True)
        audit.to_csv(proc / "audit_sheet.csv", index=False, encoding="utf-8")
        print(f"Hoja de auditoría: {len(audit)} filas -> data/processed/audit_sheet.csv")
        print(audit["motivo_auditoria"].value_counts().to_string())
    print(json.dumps(rep["resumen"]))
    return 1 if rep["resumen"]["critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
