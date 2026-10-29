# Chatbot NLP de PQRS

Clasificador **experimental** de cuatro intenciones de Peticiones, Quejas,
Reclamos y Sugerencias en salud. Esta fase se concentra en preparación,
modelado y evaluación; no implementa un chatbot de producción ni ofrece
orientación médica.

> **Datos sintéticos:** los mensajes se crean con plantillas y entidades
> ficticias. No son PQRS reales ni observaciones de pacientes. La categoría
> `wayuunaiki_basico` no ha sido validada por hablantes competentes y no permite
> afirmar comprensión fiable de wayuunaiki.

## Estado

Implementación reproducible de preparación de datos, baseline TF-IDF +
regresión logística, modelo de palabras/caracteres + LinearSVC, búsqueda
agrupada por plantilla, evaluación y pruebas. Los resultados se generan
localmente y deben interpretarse como evidencia preliminar de un corpus
sintético con plantillas repetidas.

## Estructura

```text
data/processed/       CSV sintético y registro de calidad
notebooks/            notebook de modelado ejecutable
src/pqrs_nlp/         preparación, normalización, generación y modelos
scripts/              preparación, experimentos e informe
tests/                pruebas unitarias y de integración ligeras
models/               artefacto experimental local
reports/metrics/      métricas, particiones y resultados de tuning
reports/figures/      gráficas reproducibles
```

## Requisitos e instalación

Se requiere Python 3.12 o posterior, pip y las dependencias de
`requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Reproducir

Desde la raíz del repositorio:

```bash
python scripts/prepare_dataset.py
python -m pytest
python scripts/run_experiments.py
python scripts/generate_report.py
```

El primer comando genera `data/processed/pqrs_sinteticas_limpias.csv` y
`data/processed/data_quality.json`. Los experimentos usan semilla `42`, una
partición aproximada 65/15/20 por grupos de plantillas y validación cruzada
`StratifiedGroupKFold` de cinco pliegues en entrenamiento. Los transformadores
TF-IDF se ajustan dentro de cada pipeline y pliegue. Si faltan grupos para
representar las cuatro clases, el proceso falla explícitamente en vez de
recurrir a una partición aleatoria potencialmente contaminada.

## Modelos y evaluación

* **Referencia:** `DummyClassifier(strategy="most_frequent")`.
* **Baseline:** TF-IDF de palabras (unigramas/bigramas) + regresión logística.
* **Propuesto:** TF-IDF de palabras y caracteres (`char_wb`, n-gramas 3–5) +
  `LinearSVC`. La tolerancia a erratas es una hipótesis evaluada, no una
  conclusión previa.

Se buscan `C = [0.1, 1, 10]` y `class_weight = [None, "balanced"]` para ambos
modelos entrenables, seleccionando por F1 macro en CV agrupada. La selección
final se contrasta en validación. Solo después se reentrena el modelo
seleccionado con entrenamiento + validación y se evalúa una vez con la prueba
reservada. F1 macro es primaria porque da peso igual a cada clase y evita que
la clase mayoritaria oculte un rendimiento pobre en las clases menos
representadas; también se informan accuracy, precision/recall macro, F1
ponderado y métricas por clase.

Los resultados y figuras se guardan en `reports/metrics/` y
`reports/figures/`. Se incluyen métricas agregadas y por clase, resultados por
nivel y clase, errores observados, parámetros evaluados, particiones y
matrices de confusión. El pipeline de `models/pqrs_experimental.joblib` es un
artefacto **experimental**, no apto para producción ni decisiones de salud.

## Limitaciones

* Las plantillas sintéticas pueden hacer que el rendimiento aparente sea
  superior al que se obtendría con mensajes libres de usuarios. Las
  particiones separan grupos de plantillas, pero no sustituyen una evaluación
  externa con lenguaje natural independiente.
* La diversidad léxica, los errores y la distribución de clases son simulados.
* Las expresiones `wayuunaiki_basico` son preliminares, incluyen español y no
  cuentan con revisión independiente de hablantes competentes.
* No hay validación clínica, revisión de seguridad, integración productiva,
  privacidad de datos reales ni evaluación con pacientes.

## Procedencia y siguientes pasos

El análisis exploratorio original está en `EDA_PQRS_NLP.ipynb` y se conserva
sin sobrescribir. Como ese notebook no guarda una ejecución del DataFrame, la
preparación de Fase 2 es independiente del estado interactivo de Jupyter y
genera sus propios metadatos de plantilla de manera reproducible. El proyecto
explora vocabulario del recurso público
[RLTranslator](https://github.com/Manuel-2011/RLTranslator); la referencia no
constituye validación de traducciones ni una certificación lingüística.

Próximos pasos: revisar el corpus y el reporte de calidad con el equipo,
obtener ejemplos autorizados y desidentificados si se evaluará lenguaje real,
definir evaluación externa por fuente/autor, y solicitar revisión lingüística
competente antes de cualquier trabajo de Wayuunaiki. No usar estos resultados
para atención real.
