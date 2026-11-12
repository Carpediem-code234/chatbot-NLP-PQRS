# Resumen de preparación del repositorio

## Objetivo

Organizar el repositorio y dejar documentado el entorno para iniciar posteriormente
la limpieza y preparación de datos. No se entrenaron modelos, no se cambiaron
etiquetas ni se eliminaron registros. El corpus lingüístico no se modificó ni se
descargó.

## Cambios realizados

- Se reorganizaron los archivos de datos existentes bajo `data/synthetic/raw/` y
  `data/synthetic/processed/`, conservando sus contenidos.
- Se actualizaron `scripts/train_baseline.py`, `scripts/evaluate_models.py`,
  `scripts/validate_data.py` y `tests/test_data.py` para usar la nueva ubicación
  de los datos sintéticos.
- Se fijaron las dependencias en `requirements.txt` y `pyproject.toml`, con soporte
  declarado para Python 3.12.x.
- Se añadieron `src/corpus_sources.py` y `scripts/descargar_corpus.py` para fijar
  las fuentes del corpus a commits concretos y comprobar SHA-256 antes de guardar
  archivos. La descarga requiere activación explícita.
- Se actualizó el notebook `notebooks/EDA_PQRS_NLP_CORREGIDO.ipynb`: rutas
  configurables, procedencia corregida, kernel Python 3.12 y salidas limpiadas.
- Se añadieron exclusiones para datos externos y salidas temporales en `.gitignore`,
  y un hook `nbstripout` en `.pre-commit-config.yaml`.
- Se actualizaron `README.md` y `data/README.md` con estructura, instalación,
  ejecución del notebook, rutas, procedencia y advertencias sobre licencias.

## Validaciones

- Suite de pruebas: **26 pasaron** en un entorno temporal.
- Hook `nbstripout`: pasó.
- Sintaxis de las celdas Python del notebook: pasó.
- `git diff --check`: pasó.
- Exclusiones de Git para corpus externo y salidas generadas: verificadas.

Las pruebas se ejecutaron con Python 3.14.4 porque Python 3.12 no está instalado
en este entorno. Por ello, no se ejecutó el notebook completo con la versión de
Python declarada para el proyecto.

## Pendientes y observaciones

- Instalar Python 3.12 y validar el notebook completo en ese entorno.
- Ejecutar `pre-commit install` en el entorno de desarrollo para activar el hook
  localmente.
- Confirmar con los titulares los derechos de redistribución del corpus y el
  diccionario antes de compartirlos.
- El borrado de `notebooks/EDA_PQRS_NLP.ipynb` ya estaba presente en el estado de
  Git anterior a estos cambios; la versión corregida continúa sin seguimiento.

