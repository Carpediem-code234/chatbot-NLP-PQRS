# Chatbot incluyente de PQRS en salud por WhatsApp

Chatbot de procesamiento de lenguaje natural (NLP) que recibe **Peticiones, Quejas, Reclamos y Sugerencias (PQRS)** de atención en salud por WhatsApp. Está pensado para poblaciones vulnerables de La Guajira: entiende mensajes en **lenguaje simplificado** (personas con baja alfabetización, errores ortográficos, abreviaturas) y reconoce palabras básicas en **wayuunaiki**.

> ⚠️ Proyecto académico. Los modelos se entrenaron con **datos sintéticos** y sus métricas **no representan** el desempeño con usuarios reales. El vocabulario en wayuunaiki está **pendiente de validación por hablantes nativos**.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 · Datos y EDA | Fuentes de datos, EDA del corpus español–wayuunaiki, cobertura de salud, glosario, dataset PQRS sintético y su EDA | ✅ `notebooks/01_…` |
| 2 · Desarrollo y modelado | Preprocesamiento, partición sin fuga, baseline + modelo propuesto, ajuste de hiperparámetros, evaluación | ✅ `notebooks/02_…`, `src/` |
| 2 · BETO (BERT en español) | Código listo; requiere GPU | ⏳ ejecutar en Colab |
| 3 · Integración WhatsApp Business API | — | pendiente |

## Cómo funciona

```
mensaje de WhatsApp
   │
   ├─ R0  ¿sin palabras (vacío, solo emojis)?  ──► "No entendí, ¿me cuentas qué necesitas?"
   ├─ R1  ¿alerta clínica (convulsiona, no respira…)? ──► remitir a urgencias (123)
   │
   ├─ normalización: minúsculas, sin tildes, abreviaturas (q, xq, pa), ortografía (sita→cita),
   │                 wayuunaiki→español con glosario (tottoolu→médico, ayuulii→enfermo)
   ├─ modelo M3: TF-IDF palabras ∪ n-gramas de caracteres ∪ rasgos estructurales → SVM calibrada
   │
   └─ R2  ¿confianza < τ (0,80)? ──► preguntar con botones: pedir / queja / reclamo / sugerencia
          si no ──► tipo PQRS + intención (p. ej. Reclamo · medicamento_no_entregado)
```

## Resultados (Fase 2)

F1-macro en la clasificación del **tipo de PQRS** (5 clases). Validación y test usan **plantillas nunca vistas en entrenamiento**; el test externo son 68 mensajes redactados a mano.

| Modelo | F1 val | AUC val | F1 test (10 particiones) | F1 externo (10 particiones) |
|---|---|---|---|---|
| M0 · Clase mayoritaria | 0,113 | 0,500 | — | — |
| M1 · Baseline: TF-IDF palabras + regresión logística | 0,593 | 0,826 | 0,632 ± 0,093 | 0,665 ± 0,036 |
| M2 · TF-IDF + Naive Bayes complementario | 0,555 | 0,875 | 0,660 ± 0,116 | 0,569 ± 0,049 |
| **M3 · Propuesto** | **0,657** | **0,930** | **0,677 ± 0,095** | **0,675 ± 0,030** |

- M3 es el mejor en promedio, pero la ventaja sobre el baseline es **moderada** y está dentro de la variación entre particiones.
- Con τ = 0,80, lo que el bot decide solo tiene ~92 % de exactitud en test, pero decide solo en ~36 % de los mensajes; el resto lo resuelve preguntando.
- La **intención fina** (17 clases) aún no es usable (F1 ≈ 0,3–0,45).
- **Cuello de botella: la diversidad de los datos** (≈ 70 plantillas), no el algoritmo. Detalle en `notebooks/02_modelado_PQRS.ipynb` §9.

Bitácora de experimentos: `results/experimentos.csv` · métricas: `results/resultados_modelos.csv` · figuras: `results/figuras/`.

## Estructura

```text
data/
  raw/                pqrs_salud_sintetico.csv, pqrs_test_externo.csv, glosario_salud_esp_way.csv
  external/           corpus wayuu.zip (no versionado; ver data/README.md)
notebooks/
  01_EDA_corpus_wayuu_y_PQRS.ipynb    Fase 1: fuentes, EDA, calidad de datos, glosario, dataset PQRS
  02_modelado_PQRS.ipynb              Fase 2: preprocesamiento, modelos, ajuste, evaluación, BETO
src/
  config.py           rutas, clases, hiperparámetros elegidos, umbral τ
  generar_dataset.py  generador reproducible del dataset sintético
  preprocesamiento.py normalización, glosario wayuunaiki, rasgos estructurales
  datos.py            limpieza, deduplicación, partición por plantilla sin fuga
  modelos.py          M0, M1, M2, M3
  evaluacion.py       métricas, 10 particiones, curva del umbral τ
  bot.py              reglas R0/R1/R2 + modelo
  entrenar.py         entrena, evalúa y guarda models/modelo_pqrs_m3.joblib
  predecir.py         clasifica mensajes desde la terminal
tests/                pruebas de preprocesamiento, datos (sin fuga, reproducibilidad) y modelos
results/              métricas, bitácora y figuras
models/               modelo entrenado (no versionado; se regenera)
docs/                 documento técnico
```

## Instalación y uso

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest                   # 36 pruebas
python -m src.entrenar             # entrena, evalúa y guarda el modelo (~15 s en CPU)
python -m src.predecir "no me han entregado la insulina"
python -m src.predecir             # modo interactivo
```

Para regenerar el dataset sintético: `python -m src.generar_dataset` (con la semilla por defecto produce exactamente el CSV versionado; hay una prueba que lo verifica).

Para ejecutar el notebook 01 se necesita el corpus en `data/external/wayuu.zip`. El modelo BETO (notebook 02, §8) se ejecuta en Google Colab con GPU cambiando `RUN_BERT = True`.

## Limitaciones

- Datos sintéticos generados por plantillas: vocabulario reducido y métricas optimistas.
- El test externo lo redactó el mismo equipo que diseñó las plantillas.
- Glosario wayuunaiki pequeño y sin validar; faltan términos clave (*hospital, cita, vacuna, salud*).
- La regla de urgencias R1 es una lista de palabras clave, no un modelo: no reemplaza el triage.

## Fuentes de datos

Ver [`data/README.md`](data/README.md): corpus español–wayuunaiki (Universidad de los Andes), vocabulario de enfermedades del Portal UNICISO (Kelly Johana López Epieyu y Sharol Robles Epieyu) y normativa colombiana de PQRS.
