#!/usr/bin/env python3
"""Construye el informe con los resultados producidos por run_experiments.py."""

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
OUTPUT_MD = ROOT / "reports" / "informe_tecnico_fase2.md"
OUTPUT_DOCX = ROOT / "reports" / "informe_tecnico_fase2.docx"


def _read_json(name):
    path = METRICS / name
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path.relative_to(ROOT)}. Ejecuta primero scripts/run_experiments.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(value):
    return f"{value:.3f}"


def _paragraphs(
    test, validation, tuning, quality, class_counts, level_counts, errors,
    level_table,
):
    selected = test["selected_model"]
    best = tuning.loc[tuning["is_best"]].set_index("model")
    cv_lr = best.loc["logistic_regression"]
    cv_svc = best.loc["word_char_linear_svc"]
    quality_text = (
        f"El generador produjo {quality['input_rows']} filas: se descartaron "
        f"{quality['null_text_rows_discarded']} textos nulos, "
        f"{quality['blank_text_rows_discarded']} vacíos y "
        f"{quality['duplicate_rows_discarded']} duplicados exactos o equivalentes, "
        f"quedando {quality['output_rows']} mensajes únicos."
    )
    comparison = validation.set_index("model")
    validation_text = "; ".join(
        f"{name}: F1 macro {row.f1_macro:.3f}, accuracy {row.accuracy:.3f}"
        for name, row in comparison.iterrows()
    )
    class_distribution = ", ".join(
        f"{label}: {count}" for label, count in class_counts.items()
    )
    level_distribution = ", ".join(
        f"{level}: {count}" for level, count in level_counts.items()
    )
    test_level_distribution = ", ".join(
        f"{row.nivel_lenguaje}: {row.examples}"
        for row in level_table.itertuples()
    )
    level_scores = "; ".join(
        f"{row.nivel_lenguaje}: n={row.examples}, F1 macro {_metric(row.f1_macro)}"
        + (f" ({row.interpretation_warning})"
           if pd.notna(row.interpretation_warning) else "")
        for row in level_table.itertuples()
    )
    partition = test["partition"]
    partition_detail = "; ".join(
        f"{name}: {partition['rows'][name]} mensajes "
        f"({partition['fractions_actual'][name]:.1%}), "
        f"{partition['groups'][name]} plantillas"
        for name in ("train", "validation", "test")
    )
    if errors.empty:
        error_text = (
            "No se observaron errores de clasificación en los ejemplos de prueba "
            "de esta ejecución. Este resultado describe únicamente una muestra "
            "sintética pequeña; no demuestra ausencia de errores en mensajes reales."
        )
        error_detail = "No hay ejemplos de error que enumerar en esta partición."
    else:
        samples = errors.head(5)
        error_text = (
            f"Se encontraron {len(errors)} errores entre los mensajes de prueba. "
            "La tabla de errores conserva texto, etiqueta real, predicción y nivel "
            "para inspección cualitativa; no se atribuyen causas no verificables."
        )
        error_detail = "\n".join(
            f"- “{row.texto_usuario}” — real: `{row.intencion}`, "
            f"predicha: `{row.prediccion}`, nivel: `{row.nivel_lenguaje}`."
            for row in samples.itertuples()
        )

    return [
        ("Resumen", [
            "Este avance implementa un flujo reproducible para clasificar cuatro intenciones de PQRS en salud: solicitar cita, queja por atención, reclamo de medicamentos y consulta de autorización. Se comparan un clasificador trivial, un baseline de TF-IDF de palabras con regresión logística y un modelo de TF-IDF de palabras y caracteres con LinearSVC.",
            f"El proceso de calidad aplicado a los mensajes sintéticos registró {quality['input_rows']} filas de entrada y {quality['output_rows']} mensajes válidos únicos. El modelo elegido mediante F1 macro en validación fue `{selected}`. En la prueba reservada se obtuvo accuracy {_metric(test['accuracy'])}, precision macro {_metric(test['precision_macro'])}, recall macro {_metric(test['recall_macro'])}, F1 macro {_metric(test['f1_macro'])} y F1 ponderado {_metric(test['f1_weighted'])}.",
            "Estos valores son resultados medidos sobre un corpus sintético de plantillas y no deben interpretarse como desempeño con personas usuarias. Las expresiones agrupadas como wayuunaiki básico son experimentales, incluyen español y no cuentan con validación lingüística independiente.",
        ]),
        ("1. Introducción y objetivo", [
            "Las PQRS reúnen solicitudes administrativas con propósitos diferentes. Un clasificador de intención puede servir para estudiar cómo organizar mensajes en categorías, pero no sustituye la lectura humana, la evaluación clínica ni los procedimientos institucionales. El objetivo de esta fase es completar el ciclo experimental de preparación, extracción de características, ajuste y evaluación, dejando trazabilidad de lo que se probó.",
            "El alcance excluye una interfaz de chatbot, WhatsApp, servicios externos, datos de producción, bases de datos clínicas y generación automática de respuestas. La solución es un prototipo académico de clasificación supervisada que permite comparar dos clasificadores entrenables con una referencia trivial.",
            "El resultado esperado no es demostrar utilidad clínica. Se busca comprobar que el código puede ejecutarse de manera repetible, identificar fugas previsibles por plantillas y documentar con transparencia qué limitaciones siguen abiertas.",
        ]),
        ("2. Datos, procedencia y calidad", [
            "El conjunto se genera localmente con plantillas sintéticas inspiradas en el esquema de la Fase 1. Cada fila tiene identificador, grupo de plantilla, texto original, texto con ruido potencial, intención, nivel de lenguaje, entidades descriptivas y metadatos sobre el ruido. Los valores de paciente, EPS, medicamento, fecha, especialidad y ciudad son ficticios. El CSV preparado se conserva en `data/processed/pqrs_sinteticas_limpias.csv`.",
            quality_text,
            f"La distribución de intenciones después de la limpieza fue: {class_distribution}. La distribución por nivel fue: {level_distribution}. La limpieza conserva texto fuente y metadatos, elimina textos nulos o vacíos, normaliza espacios y mayúsculas solo para detectar duplicados, y descarta repeticiones cuando la etiqueta coincide. Si un mismo texto normalizado se vincula a etiquetas incompatibles, el flujo falla en lugar de elegir una etiqueta arbitrariamente.",
            "La categoría `wayuunaiki_basico` no equivale a una traducción validada de mensajes sanitarios. El corpus usa construcciones sintéticas con elementos léxicos exploratorios y español; se requiere revisión de hablantes competentes antes de formular conclusiones lingüísticas.",
        ]),
        ("3. Preparación y control de fuga", [
            "La normalización transforma el texto a minúsculas y colapsa espacios repetidos. No elimina tildes, negaciones ni signos de interrogación, no aplica stemming o lematización y no elimina palabras vacías. Las transformaciones TF-IDF se encuentran dentro de los pipelines, por lo que se ajustan solo con los datos de entrenamiento de cada pliegue.",
            f"La partición se construye por `template_id`, separando grupos completos dentro de cada intención. La semilla es 42 y las proporciones objetivo son 65/15/20 para entrenamiento, validación y prueba. En esta corrida: {partition_detail}. La asignación busca ajustar las proporciones de filas sin compartir grupos. La información de grupos compartidos se guarda en `reports/metrics/split_metadata.json` y debe ser vacía.",
            "La búsqueda de hiperparámetros usa `StratifiedGroupKFold` de cinco pliegues sobre el conjunto de entrenamiento. Se valida que cada pliegue tenga las cuatro clases y que no comparta grupos entre entrenamiento y validación. Si faltan grupos suficientes, el proceso informa el impedimento y se detiene; no cambia silenciosamente a una partición aleatoria.",
            "La separación por plantilla reduce la fuga de frases idénticas o casi idénticas de una misma estructura. Sin embargo, las plantillas siguen definiendo tanto la gramática como vocabulario representativo. El experimento no mide la generalización a mensajes espontáneos, nuevas instituciones, autores o situaciones no incluidas en el generador.",
        ]),
        ("4. Modelos y protocolo experimental", [
            "La referencia usa `DummyClassifier(strategy='most_frequent')` para estimar el desempeño de una regla que siempre escoge la clase dominante. No se considera el baseline principal porque no aprende relaciones entre texto e intención.",
            "El baseline combina TF-IDF de unigramas y bigramas de palabras con regresión logística multiclase. TF-IDF pondera términos según su relevancia en el corpus y la regresión logística proporciona una clasificación lineal eficiente. El modelo propuesto combina TF-IDF de palabras con n-gramas de caracteres `char_wb` de longitudes 3 a 5 y `LinearSVC`; se plantea que subcadenas podrían ayudar con erratas, pero esa expectativa se somete a evaluación.",
            "Para ambos modelos entrenables se compararon `C = [0.1, 1, 10]` y `class_weight = [None, 'balanced']`, seis configuraciones por algoritmo. La métrica de selección de CV fue F1 macro. Se guardan media, desviación entre pliegues, tiempo medio de ajuste y parámetros. La selección final se revisó con F1 macro en validación, sin utilizar la prueba.",
            "Tras seleccionar el nombre del modelo, se reajustó con entrenamiento y validación y se evaluó una sola vez en el conjunto de prueba reservado. Se informan accuracy, precision y recall macro, F1 macro, F1 ponderado, métricas por clase, matriz de confusión, resultados por nivel y ejemplos de error.",
        ]),
        ("5. Resultados", [
            f"En validación con grupos de plantilla, los resultados observados fueron: {validation_text}. La elección del modelo se basó en F1 macro; no se declara ganador de antemano ni se usa accuracy como único criterio.",
            f"La configuración con mayor promedio de CV para regresión logística fue `{cv_lr['parameters']}`, con F1 macro medio {_metric(cv_lr['mean_validation_f1_macro'])}, desviación entre pliegues {_metric(cv_lr['std_validation_f1_macro'])} y ajuste medio por configuración de {_metric(cv_lr['mean_fit_time_seconds'])} s. Para el modelo de palabras/caracteres fue `{cv_svc['parameters']}`, con F1 macro medio {_metric(cv_svc['mean_validation_f1_macro'])}, desviación {_metric(cv_svc['std_validation_f1_macro'])} y ajuste medio de {_metric(cv_svc['mean_fit_time_seconds'])} s.",
            f"En la prueba final, `{selected}` obtuvo accuracy {_metric(test['accuracy'])}, precision macro {_metric(test['precision_macro'])}, recall macro {_metric(test['recall_macro'])}, F1 macro {_metric(test['f1_macro'])} y F1 ponderado {_metric(test['f1_weighted'])}. La tabla completa por clase y la matriz se guardan en `reports/metrics/test_metrics.json` y `reports/metrics/test_per_class.csv`.",
            "F1 macro promedia el F1 de las clases y evita que las clases de mayor frecuencia dominen una sola cifra global. Accuracy se presenta como información complementaria. Los resultados deben leerse junto con los soportes por clase, el reducido número de grupos y la naturaleza artificial de las frases.",
        ]),
        ("6. Resultados por nivel y análisis de errores", [
            f"Los niveles observados en la prueba fueron: {test_level_distribution}. Sus resultados medidos fueron: {level_scores}. Los archivos `reports/metrics/test_by_language.csv` y `test_by_language_per_class.csv` detallan los casos y métricas por nivel e intención. Se marca explícitamente cada subgrupo con menos de cinco observaciones por clase o menos de diez observaciones totales; sus métricas son preliminares y no se interpretan como evidencia de equidad, competencia lingüística o uso real.",
            error_text,
            error_detail,
            "La inspección cualitativa debe limitarse a ejemplos que realmente aparezcan en la tabla de errores. Si existen confusiones entre categorías, el texto, etiqueta real y predicción se ofrecen en `reports/metrics/test_errors.csv`. Una clasificación correcta sobre frases repetitivas tampoco prueba que el modelo comprenda la intención de una persona.",
            "El rendimiento podría beneficiarse de que el generador repita marcos sintácticos. La partición por grupo limita la exposición a la plantilla exacta, pero sigue habiendo vocabulario y estructuras comunes entre plantillas. Para evaluar generalización se necesitan mensajes independientes de las plantillas, particiones por fuente o autor, y una evaluación externa autorizada y desidentificada.",
        ]),
        ("7. Limitaciones, amenazas y ética", [
            "La principal amenaza a la validez es la distancia entre datos sintéticos y solicitudes reales. Las plantillas, nombres de categorías y patrones léxicos se diseñaron para representar las etiquetas; esto puede simplificar una tarea que en producción incluiría ambigüedad, contexto conversacional, abreviaturas, diversidad regional y solicitudes compuestas.",
            "La partición por `template_id` controla una forma concreta de fuga, no todas las similitudes de contenido. El tamaño reducido del corpus y de algunos grupos limita la precisión de las métricas y hace que la desviación entre pliegues sea importante. La semilla favorece la reproducibilidad, pero no elimina incertidumbre estadística.",
            "Las entidades son sintéticas y no representan información personal real. Si se incorporan datos reales en fases futuras, será necesario establecer base legal y autorización, minimización, desidentificación, control de acceso, retención y revisión institucional. Este repositorio no implementa ninguna de esas salvaguardas productivas.",
            "No existe validación por hablantes competentes de wayuunaiki ni revisión clínica. En consecuencia, el modelo y las expresiones no deben emplearse para triage, decisiones de atención, priorización de casos ni comunicación sanitaria.",
        ]),
        ("8. Conclusiones y trabajo futuro", [
            f"El trabajo entrega un flujo ejecutable, con preparación de datos, trazabilidad de plantillas, pipelines reproducibles, búsqueda agrupada, prueba reservada, métricas y pruebas automatizadas. En esta ejecución se seleccionó `{selected}` según validación y se informaron las medidas de prueba anteriores. El resultado constituye una línea base técnica, no una evaluación de producto.",
            "Los siguientes pasos recomendados son: revisar las plantillas y el balance con el equipo académico; obtener un corpus de evaluación independiente y autorizado si el alcance lo permite; definir una evaluación por fuente o hablante; estudiar intervalos de incertidumbre y calibración; y consultar hablantes competentes para todo material de Wayuunaiki. La validación externa debe preceder a cualquier afirmación de generalización.",
            "El artefacto guardado en `models/pqrs_experimental.joblib` incluye metadatos que advierten su propósito experimental. No es apto para producción ni decisiones de salud. La reproducción se realiza en el orden `prepare_dataset.py`, pruebas, `run_experiments.py` y `generate_report.py`.",
        ]),
        ("Referencias y recursos técnicos", [
            "Scikit-learn. Documentación de `Pipeline`, `TfidfVectorizer`, `GridSearchCV`, `StratifiedGroupKFold`, métricas de clasificación, `DummyClassifier`, `LogisticRegression` y `LinearSVC`. https://scikit-learn.org/stable/",
            "Manuel-2011/RLTranslator. Repositorio público explorado en el EDA para recursos léxicos. https://github.com/Manuel-2011/RLTranslator. Su inclusión no acredita la corrección de frases sanitarias ni la validez lingüística del material.",
            "Todos los resultados cuantitativos de este informe se leen de archivos CSV/JSON generados por `scripts/run_experiments.py`. No se completan tablas con resultados ficticios. La paginación del DOCX depende del motor de renderizado y debe revisarse visualmente antes de entregar.",
        ]),
    ]


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main():
    test = _read_json("test_metrics.json")
    quality_path = ROOT / "data" / "processed" / "data_quality.json"
    if not quality_path.exists():
        raise FileNotFoundError("Ejecuta primero scripts/prepare_dataset.py.")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    validation = pd.read_csv(METRICS / "validation_metrics.csv")
    tuning = pd.read_csv(METRICS / "tuning_results.csv")
    dataset = pd.read_csv(ROOT / "data" / "processed" / "pqrs_sinteticas_limpias.csv")
    errors = pd.read_csv(METRICS / "test_errors.csv")
    level_table = pd.read_csv(METRICS / "test_by_language.csv")
    level_class_table = pd.read_csv(METRICS / "test_by_language_per_class.csv")
    class_counts = dataset["intencion"].value_counts().to_dict()
    level_counts = dataset["nivel_lenguaje"].value_counts().to_dict()
    sections = _paragraphs(
        test, validation, tuning, quality, class_counts, level_counts, errors,
        level_table,
    )

    markdown = [
        "# Informe técnico de avance — Fase 2",
        "",
        "**Proyecto:** Chatbot NLP de PQRS en salud  ",
        "**Integrantes:** [Completar]  ",
        "**Docente:** [Completar]  ",
        "**Institución:** [Completar]  ",
        "**Fecha de generación:** ejecución local del informe  ",
        "**Estado:** experimento reproducible con corpus sintético; no apto para producción.",
        "",
        "## Índice",
    ]
    markdown.extend(f"{index}. [{title}](#{title.lower().replace(' ', '-')})"
                    for index, (title, _) in enumerate(sections, start=1))
    for title, paragraphs in sections:
        markdown.extend(["", f"## {title}", ""])
        markdown.extend([paragraph + "\n" for paragraph in paragraphs])

    markdown.extend([
        "",
        "## Tablas y figuras de resultados",
        "",
        "### Métricas de validación",
        _markdown_table(
            ["Modelo", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "F1 ponderado"],
            [
                [
                    row.model, _metric(row.accuracy), _metric(row.precision_macro),
                    _metric(row.recall_macro), _metric(row.f1_macro),
                    _metric(row.f1_weighted),
                ]
                for row in validation.itertuples()
            ],
        ),
        "",
        "### Métricas por intención en prueba",
        _markdown_table(
            ["Intención", "Precision", "Recall", "F1", "Soporte"],
            [
                [label, _metric(values["precision"]), _metric(values["recall"]),
                 _metric(values["f1"]), values["support"]]
                for label, values in test["per_class"].items()
            ],
        ),
        "",
        "### Métricas por nivel de lenguaje en prueba",
        _markdown_table(
            ["Nivel", "Ejemplos", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "Advertencia"],
            [
                [
                    row.nivel_lenguaje, row.examples, _metric(row.accuracy),
                    _metric(row.precision_macro), _metric(row.recall_macro),
                    _metric(row.f1_macro),
                    row.interpretation_warning if pd.notna(row.interpretation_warning) else "",
                ]
                for row in level_table.itertuples()
            ],
        ),
        "",
        "### Soporte y métricas por clase dentro de cada nivel",
        _markdown_table(
            ["Nivel", "Intención", "Soporte", "Precision", "Recall", "F1"],
            [
                [
                    row.nivel_lenguaje, row.intencion, row.support,
                    _metric(row.precision), _metric(row.recall), _metric(row.f1),
                ]
                for row in level_class_table.itertuples()
            ],
        ),
        "",
        "![Distribución de intenciones](./figures/class_distribution.png)",
        "",
        "![Comparación en validación](./figures/validation_model_comparison.png)",
        "",
        "![Matriz de confusión en prueba](./figures/test_confusion_matrix.png)",
        "",
        "![Rendimiento por intención](./figures/test_per_class.png)",
        "",
        "![Rendimiento por nivel de lenguaje](./figures/test_by_language.png)",
        "",
    ])
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(markdown), encoding="utf-8")

    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError:
        print(f"Informe Markdown generado: {OUTPUT_MD.relative_to(ROOT)}")
        print("DOCX no generado: falta python-docx.")
        return

    document = Document()
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(10)
    document.add_heading("Informe técnico de avance — Fase 2", 0)
    for line in [
        "Chatbot NLP de PQRS en salud",
        "Integrantes: [Completar]",
        "Docente: [Completar]",
        "Institución: [Completar]",
        "Documento experimental basado en datos sintéticos",
    ]:
        document.add_paragraph(line)
    document.add_page_break()

    for section_index, (title, paragraphs) in enumerate(sections):
        document.add_heading(title, level=1)
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
    document.add_page_break()
    document.add_heading("Resultados cuantitativos", level=1)
    document.add_heading("Comparación en validación", level=2)
    table = document.add_table(rows=1, cols=6)
    table.style = "Light Shading Accent 1"
    for cell, value in zip(
        table.rows[0].cells,
        ["Modelo", "Accuracy", "Prec. macro", "Recall macro", "F1 macro", "F1 ponderado"],
    ):
        cell.text = value
    for row in validation.itertuples():
        cells = table.add_row().cells
        for cell, value in zip(
            cells,
            [row.model, _metric(row.accuracy), _metric(row.precision_macro),
             _metric(row.recall_macro), _metric(row.f1_macro),
             _metric(row.f1_weighted)],
        ):
            cell.text = str(value)
    document.add_heading("Métricas por intención en prueba", level=2)
    class_table = document.add_table(rows=1, cols=5)
    class_table.style = "Light Shading Accent 1"
    for cell, value in zip(
        class_table.rows[0].cells,
        ["Intención", "Precision", "Recall", "F1", "Soporte"],
    ):
        cell.text = value
    for label, values in test["per_class"].items():
        cells = class_table.add_row().cells
        for cell, value in zip(
            cells,
            [label, _metric(values["precision"]), _metric(values["recall"]),
             _metric(values["f1"]), values["support"]],
        ):
            cell.text = str(value)
    document.add_heading("Métricas por nivel de lenguaje en prueba", level=2)
    level_doc_table = document.add_table(rows=1, cols=5)
    level_doc_table.style = "Light Shading Accent 1"
    for cell, value in zip(
        level_doc_table.rows[0].cells,
        ["Nivel", "Ejemplos", "Accuracy", "F1 macro", "Advertencia"],
    ):
        cell.text = value
    for row in level_table.itertuples():
        cells = level_doc_table.add_row().cells
        for cell, value in zip(
            cells,
            [
                row.nivel_lenguaje, row.examples, _metric(row.accuracy),
                _metric(row.f1_macro),
                row.interpretation_warning if pd.notna(row.interpretation_warning) else "",
            ],
        ):
            cell.text = str(value)
    for image_name in [
        "class_distribution.png",
        "validation_model_comparison.png",
        "test_confusion_matrix.png",
    ]:
        image_path = FIGURES / image_name
        if image_path.exists():
            document.add_page_break()
            document.add_heading(image_name.replace("_", " ").replace(".png", "").title(), level=2)
            document.add_picture(str(image_path), width=Inches(6.2))
    document.save(OUTPUT_DOCX)
    print(f"Informe Markdown generado: {OUTPUT_MD.relative_to(ROOT)}")
    print(f"Informe DOCX generado: {OUTPUT_DOCX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
