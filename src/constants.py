"""Constantes compartidas del proyecto (esquema de datos y etiquetas)."""

REQUIRED_COLUMNS = [
    "id", "text", "intent", "language", "style", "source",
    "template_id", "split", "safety_flags", "review_status", "wayuu_status",
]

INTENTS = [
    "autorizacion", "cancelar_cita", "consultar_cita", "despedida",
    "estado_pqrs", "fuera_de_alcance", "hablar_agente", "medicamentos",
    "no_entendido", "peticion_informacion", "queja_atencion",
    "reclamo_demora", "saludo", "solicitar_cita", "sugerencia",
    "urgencia_medica",
]

VALID_SPLITS = ["train", "dev", "test"]
VALID_STYLES = ["formal", "coloquial", "simplificado"]

# Clases críticas pedidas explícitamente en la especificación (secciones 5 y 10).
CRITICAL_INTENTS = ["urgencia_medica", "hablar_agente", "no_entendido"]

# Columnas que NO deben usarse como características del modelo:
# `safety_flags` se deriva de la etiqueta (ver data_validation) y el resto son metadatos.
NON_FEATURE_COLUMNS = [
    "id", "intent", "language", "style", "source", "template_id",
    "split", "safety_flags", "review_status", "wayuu_status",
]

SEED = 42
