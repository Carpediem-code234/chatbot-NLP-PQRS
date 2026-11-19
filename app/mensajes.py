"""Textos del bot en lenguaje sencillo: frases cortas, una idea por mensaje, opciones numeradas.

Los saludos en wayuunaiki ("Jamaya") están atestiguados en el corpus de la Fase 1, pero deben validarse con
hablantes nativos antes de ampliar las respuestas bilingües.
"""

NOMBRE_TIPO = {"Peticion": "una *petición* (pedir algo)", "Queja": "una *queja* (mal trato o mala atención)",
               "Reclamo": "un *reclamo* (un servicio que no te dieron o te dieron mal)",
               "Sugerencia": "una *sugerencia* (una idea para mejorar)"}

NOMBRE_INTENCION = {
    "solicitar_cita": "pedir una cita", "solicitar_autorizacion": "una autorización",
    "solicitar_documento": "pedir un documento", "solicitar_interprete": "pedir un intérprete o traductor",
    "consultar_estado": "saber cómo va tu trámite", "mal_trato": "mal trato",
    "demora_atencion": "demora en la atención", "infraestructura": "el estado del lugar de atención",
    "medicamento_no_entregado": "un medicamento que no te entregaron", "cita_incumplida": "una cita que no se cumplió",
    "cobro_indebido": "un cobro que no debían hacerte", "servicio_negado": "un servicio que te negaron",
    "mejorar_atencion": "mejorar la atención", "enfoque_intercultural": "la atención a la comunidad wayuu",
}

BIENVENIDA = ("¡Hola! 👋 Soy el asistente de *PQRS de salud*.\n"
              "Puedes contarme con tus palabras qué pasó o qué necesitas.\n"
              "Si prefieres hablar con una persona, escribe *agente*.")
BIENVENIDA_WAY = "¡Jamaya! 👋 " + BIENVENIDA.split("! 👋 ", 1)[1]

MENU = ("Quiero entenderte bien. Tu mensaje es para:\n"
        "*1* Pedir algo 📝\n*2* Poner una queja 😠\n*3* Hacer un reclamo ⚠️\n*4* Dar una sugerencia 💡\n"
        "Responde con el *número*.")

CONFIRMAR = "Entendí que es {tipo}{sobre}.\n¿Es correcto?\n*1* Sí ✅\n*2* No ❌"

PEDIR_DOCUMENTO = ("Para registrar tu solicitud necesito tu *número de documento* (cédula o tarjeta de identidad).\n"
                   "Si no lo tienes, escribe *no tengo*.")
DOCUMENTO_INVALIDO = "No encontré un número de documento. Escríbelo solo con números, o escribe *no tengo*."

RADICADO = ("✅ Listo. Registramos tu {tipo_corto}.\n"
            "Tu número de radicado es:\n*{radicado}*\n"
            "Guárdalo para preguntar cómo va. Te responderemos en máximo *15 días hábiles*.")

PEDIR_RADICADO = "Escríbeme tu número de radicado. Se ve así: *PQRS-20261119-00012*."
RADICADO_NO_ENCONTRADO = "No encontré el radicado *{radicado}*. Revisa el número o escribe *agente*."
ESTADO_RADICADO = {"radicada": "Recibida, en espera de revisión 📥", "en_tramite": "En trámite 🔄",
                   "respondida": "Respondida ✉️", "cerrada": "Cerrada ✔️"}
INFO_RADICADO = "Tu radicado *{radicado}* ({tipo_corto}):\nEstado: *{estado}*{respuesta}"

URGENCIA = ("⚠️ *Esto puede ser una emergencia.*\nLlama YA al *123* o ve al servicio de urgencias más cercano.\n"
            "Este chat no atiende emergencias.")
AGENTE = ("👤 Le avisé a un funcionario. Te escribirá por este chat en horario de atención "
          "(lunes a viernes, 7 a. m. a 5 p. m.).")
SIN_CONTENIDO = "No entendí tu mensaje 🤔. ¿Me cuentas con palabras qué necesitas?"
AUDIO = ("🎙️ Todavía no puedo escuchar notas de voz.\n"
         "Escríbeme con palabras sencillas, o escribe *agente* para hablar con una persona.")
NO_TEXTO = "Por ahora solo entiendo mensajes de texto. ¿Me cuentas con palabras qué necesitas?"
AGRADECIMIENTO = "¡Con gusto! 😊 Si necesitas algo más, escríbeme."
FUERA_DE_ALCANCE = ("Solo puedo ayudarte con *peticiones, quejas, reclamos y sugerencias de salud*.\n"
                    "¿Tienes alguna? Cuéntame.")
CANCELADO = "Listo, cancelé lo que estábamos haciendo. Cuando quieras, cuéntame qué necesitas."
OPCION_INVALIDA = "Responde solo con el número de la opción, o escribe *cancelar*."

TIPO_CORTO = {"Peticion": "petición", "Queja": "queja", "Reclamo": "reclamo", "Sugerencia": "sugerencia"}
