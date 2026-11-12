# Chatbot NLP para PQRS

Proyecto académico para explorar la clasificación de intenciones PQRS en salud.
El modelo y sus métricas actuales usan datos sintéticos pendientes de revisión
humana; no representan el desempeño con personas usuarias reales.

## Entorno

El proyecto está fijado para **Python 3.12.x**. Para crear el entorno en Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
```

En Linux/macOS, cree el entorno con `python3.12 -m venv .venv` y active
`.venv/bin/activate`.
El hook de `nbstripout` elimina salidas y contadores de ejecución antes de cada
commit.

## Estructura

```text
data/
  synthetic/
    raw/          Datos sintéticos originales y SHA256SUMS
    processed/    Derivados y hoja de auditoría
    eda/          Archivos temporales generados por el notebook (ignorado)
  external/
    wayuu_es/     Corpus local de referencia (ignorado; no modificar)
notebooks/        Análisis exploratorio
scripts/          Validación, descarga y tareas del proyecto
src/              Código reutilizable
tests/            Pruebas automatizadas
results/          Informes; results/eda/ se ignora
```

## Ejecutar el notebook

Desde la raíz del repositorio y con el entorno virtual activado:

```powershell
jupyter notebook notebooks/EDA_PQRS_NLP_CORREGIDO.ipynb
```

También puede abrir `notebooks/EDA_PQRS_NLP_CORREGIDO.ipynb` en VS Code y
seleccionar el kernel **Python 3.12 (PQRS NLP)**. Ejecute sus celdas en orden.
El notebook localiza la raíz del proyecto automáticamente si el directorio de
trabajo está dentro del repositorio. Si se ejecuta desde otro lugar, configure
`EDA_REPO_ROOT` con la ruta absoluta del repositorio antes de iniciar Jupyter.

El corpus lingüístico no es necesario para generar el análisis sintético. Si no
está en `data/external/wayuu_es/`, las secciones que dependen de él se omiten con
un aviso. Para obtener estadísticas del corpus, consulte las instrucciones en
[data/README.md](data/README.md); su licencia requiere confirmación antes de
redistribuirlo.

Para ejecución automatizada, con el directorio de trabajo en la raíz:

```powershell
jupyter nbconvert --to notebook --execute notebooks/EDA_PQRS_NLP_CORREGIDO.ipynb `
  --output eda_ejecutado.ipynb --output-dir results/eda
```

Las salidas y figuras generadas quedan en ubicaciones ignoradas por Git.

## Validar datos y pruebas

Estos comandos no entrenan modelos:

```bash
python scripts/validate_data.py
python -m pytest tests -q
```

El entrenamiento y la evaluación del baseline pertenecen a una fase posterior;
no son necesarios para preparar ni ejecutar el EDA.

Cuando se realice esa fase posterior, `evaluate_models.py` evaluará test una sola
vez por (modelo, versión de datos). El candado queda en
`results/.test_eval_*.json`; reevaluar requiere `--force` y queda registrado.

## Datos

- Los datos PQRS sintéticos versionados están en `data/synthetic/raw/` y sus
  derivados en `data/synthetic/processed/`. El generador produce 1.000 mensajes:
  600 train, 200 dev y 200 test.
- El notebook guarda las figuras y evidencias en `results/eda/`, carpeta excluida
  de Git; el CSV y JSONL sintéticos se escriben en `data/synthetic/raw/`.
- El corpus de traducción de AmericasNLP, cuando se necesita para el EDA, se
  guarda localmente en `data/external/wayuu_es/` y no se versiona.
- Para descargar los seis archivos del corpus, ejecute explícitamente:
  `python scripts/descargar_corpus.py --download`. El diccionario opcional de
  RLTranslator requiere `--include-dictionary`.

Las URLs se fijan a commits y los archivos se validan con SHA-256 antes de
escribirse. AmericasNLP es la procedencia del corpus paralelo; RLTranslator solo
provee el diccionario opcional. El corpus es material de traducción, no ejemplos
etiquetados de PQRS.

No se encontró una licencia en los árboles upstream examinados para los commits
fijados. No redistribuya el corpus ni el diccionario hasta confirmar los derechos
con sus titulares. Consulte [data/README.md](data/README.md) para el detalle de
procedencia y el estado de licencia.

## Estado

| Etapa | Estado |
|---|---|
| Validación de datos y particiones | Implementada en `scripts/validate_data.py` |
| Baseline TF-IDF + Logistic Regression | Implementado en `scripts/train_baseline.py` y `scripts/evaluate_models.py` |
| Capa de reglas de seguridad | Pendiente |
| Transformer en español | Pendiente |
| Comparación e informe final | Pendiente |
