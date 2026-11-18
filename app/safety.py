"""Capa de reglas de seguridad. Se ejecuta ANTES del modelo.

Principios: conservadora, basada en frases (no en la palabra "ayuda" sola)
y con pruebas negativas. Prefiere una falsa alarma a una urgencia ignorada.
"""
import re
import unicodedata


def normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin signos; espacios simples."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


URGENCY_PATTERNS = [
    r"\bno (puede|puedo|logra|logro) respirar\b",
    r"\bno respira\b",
    r"\bdificultad (para|al) respirar\b",
    r"\b(me|se) (esta )?ahog(o|a|ando)\b",
    r"\bdolor (fuerte |intenso )?(de|en el) pecho\b",
    r"\binfarto\b",
    r"\bderrame cerebral\b",
    r"\bdesmay(o|ado|ada|e|ar)\b",
    r"\binconsciente\b",
    r"\bse cayo\b.*\bno (se mueve|reacciona|responde|despierta)\b",
    r"\bno (reacciona|despierta)\b",
    r"\bconvulsion(es|ando)?\b",
    r"\bsangr(a|ando|e) (mucho|bastante|sin parar)\b",
    r"\bsangrado (abundante|fuerte)\b",
    r"\bhemorragia\b",
    r"\bsobredosis\b",
    r"\benvenen(o|ado|ada|amiento)\b",
    r"\bse (esta )?muriendo\b",
    r"\bquiero (morir|morirme)\b",
    r"\bquitarme la vida\b",
    r"\bsuicid",
]

_TARGET = r"(agente|asesor|asesora|humano|persona|operador|funcionario)"

AGENT_PATTERNS = [
    r"^" + _TARGET + r"$",
    # "quiero hablar con un agente", "necesito una persona", "puedo hablar con..."
    r"\b(hablar|hablarle|comunicar|comunicarme|pasar|pasarme|quiero|necesito|"
    r"requiero|prefiero|deseo) (con )?(un |una )?" + _TARGET + r"\b",
    # Mandatos: "pásame con un humano", "comuníqueme con un asesor", "transfiéreme..."
    r"\b(pasame|pasenme|comunicame|comuniqueme|conectame|conecteme|"
    r"transfiereme|transfierame) (con )?(un |una )?" + _TARGET + r"\b",
    r"\batencion humana\b",
]

_URGENCY = [re.compile(p) for p in URGENCY_PATTERNS]
_AGENT = [re.compile(p) for p in AGENT_PATTERNS]


def check_rules(text: str) -> str | None:
    """Devuelve 'urgencia_medica', 'hablar_agente' o None si no aplica ninguna regla."""
    t = normalize(text)
    if any(p.search(t) for p in _URGENCY):
        return "urgencia_medica"
    if any(p.search(t) for p in _AGENT):
        return "hablar_agente"
    return None