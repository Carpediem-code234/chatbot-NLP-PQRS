# Datos del proyecto

`processed/pqrs_sinteticas_limpias.csv` se genera localmente con
`python scripts/prepare_dataset.py`. Su fuente es un generador reproducible de
plantillas del repositorio; **no** son solicitudes reales ni datos de pacientes.
La semilla por defecto es `42`. El proceso introduce deliberadamente erratas,
registros vacíos o nulos y duplicados para poder medir su tratamiento.

El CSV conserva:

| Columna | Uso |
|---|---|
| `id_registro` | Trazabilidad sintética; no se usa como característica. |
| `template_id` | Grupo para separar plantillas entre particiones; no se usa para predecir. |
| `texto_original` | Texto antes de la inyección sintética de ruido. |
| `texto_usuario` | Texto de entrada al clasificador. |
| `texto_normalizado` | Mayúsculas y espacios normalizados de forma conservadora. |
| `intencion` | Etiqueta objetivo con cuatro clases. |
| `nivel_lenguaje` | Metadato para análisis posterior, nunca entrada del clasificador. |
| `entidades_etiquetadas` | Metadato descriptivo, nunca entrada del clasificador. |
| `ruido_introducido` | Indicador sintético de la modificación ortográfica. |
| `tipo_ruido` | Tipo de ruido o causa de invalidez registrada. |

La preparación descarta textos nulos/vacíos, detecta y rechaza mensajes
normalizados con etiquetas incompatibles, y elimina duplicados conservando la
primera fila. El recuento por causa se escribe en
`processed/data_quality.json`. El generador registra los identificadores de
plantilla de origen; no se reconstruyen metadatos de filas antiguas.

`wayuunaiki_basico` es una categoría experimental: las expresiones mezclan
elementos léxicos del material exploratorio con español y no cuentan con
validación lingüística independiente. No representan traducciones sanitarias
validadas ni permiten afirmar que el sistema atienda en wayuunaiki.
