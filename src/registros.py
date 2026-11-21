"""Registros de entrenamiento agregados a mano (por WhatsApp con #nuevo o por cargue masivo).

Se guardan en data/raw/registros_nuevos.csv, separados del dataset sintético, con el mismo esquema.
El entrenamiento (src/datos.py) los suma automáticamente. Cada registro es su propia plantilla
(plantilla_id = manual_<id>), así que la partición train/val/test por plantilla sigue sin fuga.
"""
from datetime import datetime

import pandas as pd

from src.config import DATASET_PQRS, REGISTROS_NUEVOS, TEST_EXTERNO
from src.preprocesamiento import cargar_way2esp, normalizar

COLUMNAS = ["id_msg", "texto", "tipo_pqrs", "intencion", "registro", "wayuunaiki", "plantilla_id", "fuente"]

# Intenciones válidas por tipo, con el nombre que se muestra en el formulario
CATALOGO = {
    "Peticion": {"solicitar_cita": "Pedir una cita", "solicitar_autorizacion": "Pedir una autorización",
                 "solicitar_documento": "Pedir un documento", "solicitar_interprete": "Pedir intérprete",
                 "consultar_estado": "Consultar un trámite", "solicitar_remision": "Pedir remisión",
                 "afiliacion_eps": "Afiliación / cambio de EPS", "solicitar_transporte": "Pedir transporte"},
    "Queja": {"mal_trato": "Mal trato", "demora_atencion": "Demora en la atención",
              "infraestructura": "Estado del lugar"},
    "Reclamo": {"medicamento_no_entregado": "Medicamento no entregado", "cita_incumplida": "Cita incumplida",
                "cobro_indebido": "Cobro indebido", "servicio_negado": "Servicio negado",
                "demora_autorizacion": "Autorización demorada",
                "procedimiento_no_realizado": "Procedimiento no realizado"},
    "Sugerencia": {"mejorar_atencion": "Mejorar la atención", "enfoque_intercultural": "Enfoque intercultural"},
    "Otro": {"saludo": "Saludo", "agradecimiento": "Agradecimiento", "felicitacion": "Felicitación",
             "fuera_de_dominio": "Fuera de tema"},
}
REGISTROS = ["formal", "coloquial", "simplificado"]


def tiene_wayuunaiki(texto: str) -> bool:
    way = cargar_way2esp()
    return any(w.strip(".,!?¡¿").lower() in way or w.lower().startswith("jamaya") for w in texto.split())


def _textos_existentes() -> set:
    vistos = set()
    for ruta in (DATASET_PQRS, REGISTROS_NUEVOS, TEST_EXTERNO):
        if ruta.exists():
            vistos.update(normalizar(t) for t in pd.read_csv(ruta, keep_default_na=False).texto)
    return vistos


def validar(texto: str, tipo: str, intencion: str, registro: str = "coloquial", existentes=None) -> str | None:
    """Devuelve None si el registro es válido; si no, el motivo del rechazo."""
    texto = (texto or "").strip()
    if not normalizar(texto):
        return "el mensaje está vacío"
    if tipo not in CATALOGO:
        return f"tipo inválido '{tipo}' (usa: {', '.join(CATALOGO)})"
    if intencion not in CATALOGO[tipo]:
        return f"la intención '{intencion}' no corresponde al tipo {tipo}"
    if registro not in REGISTROS:
        return f"registro inválido '{registro}' (usa: {', '.join(REGISTROS)})"
    existentes = _textos_existentes() if existentes is None else existentes
    if normalizar(texto) in existentes:
        return "ese mensaje ya existe en los datos (o en el test externo)"
    return None


def agregar(filas: list[dict], fuente: str) -> pd.DataFrame:
    """Agrega registros YA validados a registros_nuevos.csv y devuelve las filas guardadas."""
    previo = pd.read_csv(REGISTROS_NUEVOS, keep_default_na=False) if REGISTROS_NUEVOS.exists() else None
    inicio = 0 if previo is None else len(previo)
    sello = datetime.now().strftime("%Y%m%d")
    nuevas = []
    for i, f in enumerate(filas, start=inicio + 1):
        nuevas.append({"id_msg": f"R{i:05d}", "texto": f["texto"].strip(), "tipo_pqrs": f["tipo_pqrs"],
                       "intencion": f["intencion"], "registro": f.get("registro") or "coloquial",
                       "wayuunaiki": tiene_wayuunaiki(f["texto"]), "plantilla_id": f"manual_{sello}_{i:05d}",
                       "fuente": fuente})
    nuevas = pd.DataFrame(nuevas, columns=COLUMNAS)
    total = nuevas if previo is None else pd.concat([previo, nuevas], ignore_index=True)
    total.to_csv(REGISTROS_NUEVOS, index=False)
    return nuevas
