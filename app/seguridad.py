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
    r"\bquiero (morir|morirme)\b", r"\bquitarme la vida\b", r"\bsuicid",
    # ── Añadidas: casos que antes NO se detectaban ──
    r"\bduele\w* (mucho |muy |demasiado |fuerte |duro )*el pecho\b",                 # "me duele mucho el pecho"
    r"\b(me|se) (quiero|quiere|voy a|va a) matar\b", r"\bmatarme\b",                 # ideación suicida
    r"\bno (quiero|quiere) (vivir|seguir viviendo)\b", r"\bacabar con mi vida\b",
    r"\bveneno\b", r"\bintoxica\w*",                                               # intoxicación
    r"\bse (tomo|trago|bebio) (el |la |un |una )?(cloro|creolina|gramoxone|raticida|matarratas)\b",
    r"\bdesangr\w*",                                                                 # sangrado
    r"\b(parto|embarazada|embarazo)\b.*\bsangr\w*", r"\bsangr\w*.*\b(parto|embarazada|embarazo)\b",
    r"\bse (puso|esta poniendo|esta) (morado|morada|azul)\b", r"\blabios morados\b",  # falta de oxígeno
    r"\b(lo|la|me|le) (mordio|pico) (un |una )?(culebra|serpiente|vibora|cascabel|alacran|escorpion)\b",
    r"\bse quemo\b", r"\bquemadura\w*",                                             # quemaduras
    r"\baccidente\b",                                                                # accidentes
    r"\b(esta|sigue) (muy )?grave\b",
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
