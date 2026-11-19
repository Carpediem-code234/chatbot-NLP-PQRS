"""Reglas de seguridad: se ejecutan ANTES del modelo y en cualquier punto de la conversación.

Basado en `app/safety.py` de la rama Whatsapp-Api-integracion del equipo, con estos cambios:
- se aplica sobre el texto ya normalizado por `src.preprocesamiento.normalizar` (corrige ortografía,
  abreviaturas y traduce términos en wayuunaiki), así "se esta ahogando", "SE ESTA AOGANDO" o mensajes con
  palabras wayuu pasan por las mismas reglas;
- se añaden variantes y errores de escritura frecuentes ("convulsiona", "aogando", "desmallo", "mordio una culebra/serpiente", "no despierta") y la petición de agente
  en lenguaje sencillo ("una persona de verdad").

Principio: prefiere una falsa alarma a una urgencia ignorada. NO reemplaza el triage médico.
"""
import re

from src.preprocesamiento import normalizar

URGENCIA = [
    r"\bno (puede|puedo|logra|logro|pude) respirar\b", r"\bno respira\b", r"\bdificultad (para|al) respirar\b",
    r"\b(me|se) (esta )?a(h)?og(o|a|ando)\b", r"\bdolor (muy )?(fuerte |intenso )?(de|en el) pecho\b",
    r"\binfarto\b", r"\bderrame cerebral\b", r"\bdesma(y|ll)(o|ado|ada|e|ar)\b", r"\binconsciente\b",
    r"\bse cayo\b.*\bno (se mueve|reacciona|responde|despierta)\b", r"\bno (reacciona|despierta)\b",
    r"\bconvulsion(es|ando)?\b", r"\bconvulsiona(ndo)?\b", r"\bcombulsi\w*", r"\bconbulsi\w*",
    r"\bsangr(a|ando|e) (mucho|bastante|sin parar)\b", r"\bsangrado (abundante|fuerte)\b", r"\b(h)?emorragia\b",
    r"\bsobredosis\b", r"\benvenen(o|ado|ada|amiento)\b", r"\bse (esta )?muriendo\b",
    r"\b(lo|la|me) (mordio|pico) una? (culebra|serpiente|vibora|cascabel)\b",
    r"\bquiero (morir|morirme)\b", r"\bquitarme la vida\b", r"\bsuicid",
]

_DESTINO = r"(agente|asesor|asesora|humano|persona|operador|operadora|funcionario|funcionaria)"
AGENTE = [
    r"^" + _DESTINO + r"$",
    r"\b(hablar|hablarle|comunicar|comunicarme|pasar|pasarme|quiero|necesito|requiero|prefiero|deseo) "
    r"(con )?(un |una )?" + _DESTINO + r"\b",
    r"\b(pasame|pasenme|comunicame|comuniqueme|conectame|conecteme|transfiereme|transfierame) (con )?(un |una )?"
    + _DESTINO + r"\b",
    r"\batencion humana\b", r"\bpersona de verdad\b",
]

_URG = [re.compile(p) for p in URGENCIA]
_AGE = [re.compile(p) for p in AGENTE]


def revisar(texto: str) -> str | None:
    """Devuelve 'urgencia', 'agente' o None."""
    t = normalizar(texto)
    if any(p.search(t) for p in _URG):
        return "urgencia"
    if any(p.search(t) for p in _AGE):
        return "agente"
    return None
