# Fase 2 — Modelado de intenciones PQRS (Grupo 07)

Estado por etapa (solo se marca lo ejecutado y verificado):

| Etapa | Estado |
|---|---|
| 1. Validación de datos y particiones | Hecha: `scripts/validate_data.py`, informe en `results/data_validation/` |
| 2. Baseline TF-IDF + Logistic Regression | Hecha: `scripts/train_baseline.py`, `scripts/evaluate_models.py` |
| 3. Capa de reglas de seguridad | Pendiente (`src/safety_rules.py`, `tests/test_safety.py`) |
| 4. Transformer en español | Pendiente (requiere torch/transformers y descarga del modelo) |
| 5. Comparación, informe 8–12 págs., notebook | Pendiente |

## Ejecución
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/validate_data.py        # informe de calidad + hoja de auditoría
python scripts/train_baseline.py       # NO carga el test; selecciona con CV agrupado y dev
python scripts/evaluate_models.py --models baseline   # evaluación única en test (candado)
python -m pytest tests -q
```

## Reglas de uso del test
`evaluate_models.py` evalúa el test una sola vez por (modelo, versión de datos); el candado queda en `results/.test_eval_*.json`. Reevaluar requiere `--force` y queda registrado.

## Advertencia
Todas las métricas provienen de datos **sintéticos pendientes de revisión humana**. No representan el desempeño con usuarios reales.
