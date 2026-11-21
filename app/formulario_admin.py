"""Formulario por WhatsApp para que un ADMINISTRADOR agregue registros de entrenamiento.

Se activa escribiendo  #nuevo  desde un número listado en ADMIN_WA (.env). Pasos:
    admin_texto → admin_tipo → admin_intencion → admin_registro → admin_confirmar
En cualquier paso, "cancelar" sale del formulario. El registro se guarda en data/raw/registros_nuevos.csv
y se usa en el próximo entrenamiento (python -m src.entrenar).
"""
from src import registros as R
from src.preprocesamiento import normalizar

COMANDO = "#nuevo"
TIPOS = list(R.CATALOGO)                      # Peticion, Queja, Reclamo, Sugerencia, Otro
NOMBRE_TIPO = {"Peticion": "Petición", "Queja": "Queja", "Reclamo": "Reclamo", "Sugerencia": "Sugerencia",
               "Otro": "Otro (saludo, gracias, felicitación, fuera de tema)"}
CANCELAR = {"cancelar", "salir", "0"}


def _menu(titulo, opciones):
    return titulo + "\n" + "\n".join(f"*{i}* {o}" for i, o in enumerate(opciones, 1)) + "\n_(cancelar para salir)_"


def _elegir(texto, n):
    t = texto.strip()
    return int(t) - 1 if t.isdigit() and 1 <= int(t) <= n else None


class FormularioAdmin:
    def __init__(self, almacen, admins: set[str]):
        self.db, self.admins = almacen, {a for a in admins if a}
        self.al_guardar = None          # función(tel) -> str | None; la define Dialogo para reentrenar

    def es_admin(self, telefono: str) -> bool:
        return "".join(c for c in telefono if c.isdigit()) in self.admins

    def activa(self, telefono, texto, estado) -> bool:
        """¿Este mensaje lo maneja el formulario? (comando #nuevo o formulario en curso, solo admins)."""
        return self.es_admin(telefono) and (estado.startswith("admin_") or texto.strip().lower() == COMANDO)

    def responder(self, tel, texto, estado, datos) -> list[str]:
        if texto.strip().lower() == COMANDO:
            self.db.guardar_conversacion(tel, "admin_texto", {})
            return ["📝 *Nuevo registro de entrenamiento*\nEscribe el MENSAJE tal como lo escribiría un usuario."
                    "\n_(cancelar para salir)_"]
        if normalizar(texto) in CANCELAR:
            self.db.guardar_conversacion(tel, "inicio")
            return ["❌ Registro cancelado. No se guardó nada."]
        return getattr(self, "_" + estado)(tel, texto, datos)

    def _admin_texto(self, tel, texto, datos):
        motivo = R.validar(texto, "Otro", "saludo")        # aquí solo importa: vacío o repetido
        if motivo:
            return [f"⚠️ No se puede usar: {motivo}. Escribe otro mensaje."]
        self.db.guardar_conversacion(tel, "admin_tipo", {"texto": texto.strip()})
        return [_menu("¿Qué *TIPO* es?", [NOMBRE_TIPO[t] for t in TIPOS])]

    def _admin_tipo(self, tel, texto, datos):
        i = _elegir(texto, len(TIPOS))
        if i is None:
            return [_menu("Responde con un número. ¿Qué *TIPO* es?", [NOMBRE_TIPO[t] for t in TIPOS])]
        datos["tipo_pqrs"] = TIPOS[i]
        self.db.guardar_conversacion(tel, "admin_intencion", datos)
        return [_menu("¿Qué *INTENCIÓN*?", list(R.CATALOGO[TIPOS[i]].values()))]

    def _admin_intencion(self, tel, texto, datos):
        opciones = list(R.CATALOGO[datos["tipo_pqrs"]])
        i = _elegir(texto, len(opciones))
        if i is None:
            return [_menu("Responde con un número. ¿Qué *INTENCIÓN*?", list(R.CATALOGO[datos["tipo_pqrs"]].values()))]
        datos["intencion"] = opciones[i]
        self.db.guardar_conversacion(tel, "admin_registro", datos)
        return [_menu("¿Cómo está *ESCRITO*?", ["Formal", "Coloquial", "Simplificado (errores, baja alfabetización)"])]

    def _admin_registro(self, tel, texto, datos):
        i = _elegir(texto, 3)
        if i is None:
            return [_menu("Responde con un número. ¿Cómo está *ESCRITO*?", ["Formal", "Coloquial", "Simplificado"])]
        datos["registro"] = R.REGISTROS[i]
        self.db.guardar_conversacion(tel, "admin_confirmar", datos)
        way = "Sí" if R.tiene_wayuunaiki(datos["texto"]) else "No"
        return [f"📋 *Resumen*\n• Mensaje: {datos['texto']}\n• Tipo: {NOMBRE_TIPO[datos['tipo_pqrs']].split(' (')[0]}"
                f"\n• Intención: {R.CATALOGO[datos['tipo_pqrs']][datos['intencion']]}\n• Escrito: {datos['registro']}"
                f"\n• Wayuunaiki: {way} (detectado)\n\n¿Guardar?\n*1* Sí ✅\n*2* No ❌"]

    def _admin_confirmar(self, tel, texto, datos):
        t = normalizar(texto)
        if t in {"1", "si", "aa"}:
            motivo = R.validar(datos["texto"], datos["tipo_pqrs"], datos["intencion"], datos["registro"])
            self.db.guardar_conversacion(tel, "inicio")
            if motivo:
                return [f"⚠️ No se guardó: {motivo}."]
            fila = R.agregar([datos], fuente="manual_whatsapp").iloc[0]
            aviso = self.al_guardar(tel) if self.al_guardar else None
            if aviso:
                return [f"✅ Guardado como *{fila.id_msg}*.", aviso]
            return [f"✅ Guardado como *{fila.id_msg}*. Se usará en el próximo entrenamiento "
                    f"(python -m src.entrenar).\nEscribe *{COMANDO}* para agregar otro."]
        if t in {"2", "no", "nnojo"}:
            self.db.guardar_conversacion(tel, "inicio")
            return ["❌ Registro descartado."]
        return ["Responde *1* para guardar o *2* para descartar."]
