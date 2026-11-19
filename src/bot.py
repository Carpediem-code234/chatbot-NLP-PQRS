"""Capa de decisión del chatbot: reglas de seguridad + modelo.

R0  mensaje sin palabras            → pedir que reformule
R1  palabras de alerta clínica      → remitir a urgencias (no se clasifica como PQRS)
R2  confianza del modelo < τ        → preguntar con botones en vez de adivinar
"""
import re

import joblib
import pandas as pd

from src.config import MODELO
from src.preprocesamiento import normalizar, preparar

ALERTA = re.compile(r"no respira|convuls|sangra(do|ndo)|inconscien|desmay|suicid|se esta muriendo|envenen|"
                    r"mordio una culebra")

RESPUESTA_R0 = "No entendí tu mensaje. ¿Me cuentas qué necesitas? Puedes escribir o enviar una nota de voz."
RESPUESTA_R1 = "⚠️ Esto puede ser una urgencia. Llama ya al 123 o acude al servicio de urgencias más cercano."
RESPUESTA_R2 = "¿Tu mensaje es para: 1️⃣ pedir algo, 2️⃣ poner una queja, 3️⃣ un reclamo o 4️⃣ dar una sugerencia?"


def regla_alerta(texto_norm: str) -> bool:
    return bool(ALERTA.search(texto_norm))


class ClasificadorPQRS:
    def __init__(self, ruta=MODELO):
        art = joblib.load(ruta)
        self.m_tipo, self.m_int, self.tau = art["tipo"], art["intencion"], art["tau"]

    def clasificar(self, texto: str) -> dict:
        norm = normalizar(texto)
        if not norm.strip():
            return {"accion": "R0_reformular", "respuesta": RESPUESTA_R0}
        if regla_alerta(norm):
            return {"accion": "R1_urgencia", "respuesta": RESPUESTA_R1}
        fila = preparar(pd.DataFrame({"texto": [texto]}))
        p = self.m_tipo.predict_proba(fila)[0]
        tipo, conf = self.m_tipo.classes_[p.argmax()], float(p.max())
        if conf < self.tau:
            return {"accion": "R2_preguntar", "respuesta": RESPUESTA_R2, "tipo_probable": tipo,
                    "confianza": round(conf, 2)}
        return {"accion": "clasificado", "tipo_pqrs": tipo, "intencion": self.m_int.predict(fila)[0],
                "confianza": round(conf, 2)}
