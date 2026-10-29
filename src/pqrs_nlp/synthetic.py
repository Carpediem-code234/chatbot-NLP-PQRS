"""Generación reproducible del corpus sintético de PQRS."""

import random
import re

import pandas as pd


INTENT_TEMPLATES = {
    "solicitar_cita": {
        "formal": [
            "Solicito una cita de {especialidad} para {nombre}, afiliado a {eps}, antes del {fecha}.",
            "Solicito formalmente una cita médica en {especialidad} en {ciudad}; mi EPS es {eps}.",
            "Por medio de este mensaje solicito agendar cita de {especialidad} para {nombre} en {ciudad}.",
        ],
        "simplificado": [
            "Hola, necesito una cita con el médico de {especialidad} para {nombre}.",
            "Quiero pedir cita en {ciudad}, es para {especialidad}, por favor.",
            "Necesito cita pronto, es para {nombre}, con {especialidad}.",
        ],
        "wayuunaiki_basico": [
            "{saludo}, {yo} {nombre}. Necesito cita con el médico de {especialidad}, EPS {eps}.",
            "{saludo}, {yo} {nombre}. Por favor, una cita de {especialidad} en {ciudad}, EPS {eps}.",
        ],
    },
    "queja_atencion": {
        "formal": [
            "Presento una queja formal por la atención recibida el {fecha} en {ciudad} por parte de {eps}.",
            "Deseo dejar constancia de mi inconformidad con el trato del personal de {eps} el {fecha}.",
            "Presento queja por la deficiente atención prestada a {nombre} en {ciudad} por {eps}.",
        ],
        "simplificado": [
            "Quiero quejarme, me trataron mal en la EPS {eps} el día {fecha}.",
            "No me gustó cómo me atendieron en {ciudad}, fue muy grosero el trato.",
            "Me atendieron muy mal en {eps}, quiero poner una queja.",
        ],
        "wayuunaiki_basico": [
            "{saludo}, {yo} {nombre}. Mala atención en EPS {eps} el {fecha}. Por favor, ayúdenme.",
            "{saludo}, {yo} {nombre}. No me gustó la atención en {ciudad}, EPS {eps}, el {fecha}.",
        ],
    },
    "reclamo_medicamentos": {
        "formal": [
            "Formulo reclamo por la entrega tardía del medicamento {medicamento} para {nombre}, afiliado a {eps}.",
            "Solicito la entrega de {medicamento}, que no ha sido dispensado desde el {fecha} por {eps}.",
            "Presento reclamo ante {eps} por falta de suministro de {medicamento} al paciente {nombre}.",
        ],
        "simplificado": [
            "No me han dado el medicamento {medicamento}, lo necesito ya.",
            "Hace días pedí {medicamento} y en {eps} todavía no me lo entregan.",
            "Necesito que me entreguen {medicamento}, llevo esperando desde {fecha}.",
        ],
        "wayuunaiki_basico": [
            "{saludo}, {yo} {nombre}. Necesito el medicamento {medicamento}, EPS {eps}, desde {fecha}.",
            "{saludo}, {yo} {nombre}. No me dan {medicamento} en EPS {eps} desde {fecha}.",
        ],
    },
    "consultar_autorizacion": {
        "formal": [
            "Solicito información sobre el estado de la autorización para {nombre} ante {eps}.",
            "Requiero conocer el estado de la autorización de {especialidad} tramitada el {fecha} en {eps}.",
            "Agradezco informar el estado de la autorización solicitada por {nombre} en {ciudad}.",
        ],
        "simplificado": [
            "¿Cómo va mi autorización para {especialidad}? La pedí en {eps}.",
            "Quiero saber si ya está lista mi autorización, {nombre} es el paciente.",
            "¿Ya salió la autorización que pedí el {fecha} en {eps}?",
        ],
        "wayuunaiki_basico": [
            "{saludo}, {yo} {nombre}. Por favor, ¿autorización de {especialidad}, EPS {eps}?",
            "{saludo}, {yo} {nombre}. Gracias, ¿está lista mi autorización de {especialidad} en {eps}?",
        ],
    },
}

NOMBRES = ["María", "José", "Carmen", "Luis", "Ana", "Pedro", "Rosa", "Jairo"]
MEDICAMENTOS = ["Losartán", "Metformina", "Acetaminofén", "Insulina", "Omeprazol"]
EPS = ["SaludTotal", "ViveEPS", "MediSalud", "EPS Cesar Salud"]
ESPECIALIDADES = ["medicina general", "pediatría", "cardiología", "odontología"]
CIUDADES = ["Valledupar", "Riohacha", "Maicao", "Uribia"]
CLASS_WEIGHTS = {
    "solicitar_cita": 0.40,
    "queja_atencion": 0.15,
    "reclamo_medicamentos": 0.30,
    "consultar_autorizacion": 0.15,
}
NOISE_RATE = 0.12
SEED = 42


def _random_values(rng):
    return {
        "nombre": rng.choice(NOMBRES),
        "medicamento": rng.choice(MEDICAMENTOS),
        "eps": rng.choice(EPS),
        "especialidad": rng.choice(ESPECIALIDADES),
        "ciudad": rng.choice(CIUDADES),
        "fecha": f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025",
        "saludo": rng.choice(["Jamaan", "Jamaya"]),
        "yo": "taya",
    }


def _introduce_noise(text, rng):
    without_accents = (
        text.translate(str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU"))
    )
    words = without_accents.split()
    if len(words) > 3:
        index = rng.randrange(len(words))
        if len(words[index]) > 1:
            words[index] = words[index][0] * 2 + words[index][1:]
    return " ".join(words)


def _entity_text(template, values):
    tags = {
        "nombre": "PACIENTE",
        "medicamento": "MEDICAMENTO",
        "eps": "EPS",
        "especialidad": "ESPECIALIDAD",
        "ciudad": "CIUDAD",
        "fecha": "FECHA",
    }
    return "; ".join(
        f"{tags[key]}:{values[key]}"
        for key in re.findall(r"\{(\w+)\}", template)
        if key in tags
    ) or "SIN_ENTIDADES"


def generate_synthetic_dataframe(seed=SEED, n_records=240):
    """Genera texto, etiqueta, nivel, grupo de plantilla y trazabilidad."""
    if n_records < len(INTENT_TEMPLATES) * 8:
        raise ValueError("n_records debe permitir al menos una fila por plantilla.")
    rng = random.Random(seed)
    templates = [
        (intent, level, index, template)
        for intent, levels in INTENT_TEMPLATES.items()
        for level, items in levels.items()
        for index, template in enumerate(items)
    ]
    records = []
    counts = {intent: 0 for intent in INTENT_TEMPLATES}

    for intent, level, index, template in templates:
        values = _random_values(rng)
        text = template.format(**values)
        template_id = f"{intent}__{level}__{index:02d}"
        records.append({
            "id_registro": f"pqrs-{len(records) + 1:04d}",
            "template_id": template_id,
            "texto_original": text,
            "texto_usuario": text,
            "intencion": intent,
            "nivel_lenguaje": level,
            "entidades_etiquetadas": _entity_text(template, values),
            "ruido_introducido": False,
            "tipo_ruido": "sin_ruido",
        })
        counts[intent] += 1

    while len(records) < n_records:
        intent = rng.choices(
            list(CLASS_WEIGHTS), weights=list(CLASS_WEIGHTS.values()), k=1
        )[0]
        level = rng.choices(
            ["formal", "simplificado", "wayuunaiki_basico"],
            weights=[0.55, 0.30, 0.15],
            k=1,
        )[0]
        options = INTENT_TEMPLATES[intent][level]
        index = rng.randrange(len(options))
        template = options[index]
        values = _random_values(rng)
        text = template.format(**values)
        template_id = f"{intent}__{level}__{index:02d}"
        records.append({
            "id_registro": f"pqrs-{len(records) + 1:04d}",
            "template_id": template_id,
            "texto_original": text,
            "texto_usuario": text,
            "intencion": intent,
            "nivel_lenguaje": level,
            "entidades_etiquetadas": _entity_text(template, values),
            "ruido_introducido": False,
            "tipo_ruido": "sin_ruido",
        })
        counts[intent] += 1

    noise_indices = rng.sample(range(len(records)), int(len(records) * NOISE_RATE))
    for index in noise_indices:
        records[index]["texto_usuario"] = _introduce_noise(
            records[index]["texto_original"], rng
        )
        records[index]["ruido_introducido"] = True
        records[index]["tipo_ruido"] = "sin_tildes_y_letra_duplicada"

    duplicate_indices = rng.sample(range(len(records)), 8)
    for index in duplicate_indices:
        duplicate = records[index].copy()
        duplicate["id_registro"] = f"pqrs-{len(records) + 1:04d}"
        records.append(duplicate)

    for offset in range(6):
        row = records[rng.randrange(n_records)].copy()
        row["id_registro"] = f"pqrs-{len(records) + 1:04d}"
        row["texto_usuario"] = None
        row["tipo_ruido"] = "nulo_insertado"
        records.append(row)
    for _ in range(5):
        row = records[rng.randrange(n_records)].copy()
        row["id_registro"] = f"pqrs-{len(records) + 1:04d}"
        row["texto_usuario"] = ""
        row["tipo_ruido"] = "vacio_insertado"
        records.append(row)

    return pd.DataFrame(records)
