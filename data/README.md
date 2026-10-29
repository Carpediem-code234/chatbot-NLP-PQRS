# Datos

- `raw/pqrs_sintetico.csv` y `raw/pqrs_sintetico.jsonl`: copia **sin modificar** de la Fase 1 (verificar con `sha256sum -c SHA256SUMS`). El CSV y el JSONL son idénticos (comprobado en `tests/test_data.py`).
- `processed/audit_sheet.csv`: hoja de auditoría humana (290 filas: test completo, todas las `urgencia_medica` y `hablar_agente`, y muestra estratificada de train/dev con semilla 42). Las columnas `etiqueta_revisor`, `acuerdo_etiqueta`, `notas_revisor` y `revisor` están **vacías a propósito**: las completa una persona.
- Todos los registros siguen en `review_status = pendiente_revision_humana`. Ninguna transformación modifica el texto original.
- Los datos son **sintéticos y ficticios**; no representan PQRS reales.
- El corpus español–wayuunaiki (AmericasNLP/RLTranslator) es un corpus de traducción y **no se usa** como dato supervisado de intención.
- No usar `safety_flags`, `style`, `template_id`, `id` como características: `safety_flags` identifica sin ambigüedad 3 intenciones.
