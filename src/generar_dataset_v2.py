"""Generador AMPLIADO del dataset sintético de PQRS (v2): 23 intenciones y mucha más variedad.

Combina, por intención, NÚCLEOS (la situación) con piezas intercambiables (contexto inicial, cierre,
personas, lugares, servicios, medicamentos...) y añade FRASES LIBRES escritas completas. Cada frase base
se expande en 3 variantes (formal, coloquial, simplificado) que comparten `plantilla_id`, para que la
partición train/val/test por plantilla siga sin fuga.

Uso:
    python -m src.generar_dataset_v2            # escribe data/raw/pqrs_salud_ampliado.csv (original + nuevos)
    python -m src.generar_dataset_v2 --por-intencion 140 --seed 7

Los datos siguen siendo SINTÉTICOS: deben revisarse por personas y reportarse como tales.
"""
import argparse
import random
import re

import pandas as pd

from src import generar_dataset as gd
from src.config import DATA_RAW, DATASET_PQRS, TEST_EXTERNO
from src.preprocesamiento import normalizar

SALIDA = DATASET_PQRS   # v1 (3.165 filas, intactas) + filas nuevas v2

S = {k: list(v) for k, v in gd.SLOTS.items()}
S["servicio"] += ["cardiologia", "ortopedia", "oftalmologia", "dermatologia", "fisioterapia", "neurologia",
                  "urologia", "optometria", "medicina familiar", "terapia respiratoria"]
S["lugar"] += ["San Juan del Cesar", "Hatonuevo", "Barrancas", "Villanueva", "Distraccion", "Cuestecitas",
               "Paraguachon", "Cabo de la Vela", "Puerto Bolivar", "Mayapo"]
S["medicamento"] += ["atorvastatina", "levotiroxina", "omeprazol", "glibenclamida", "carbamazepina",
                     "sulfato ferroso", "prednisolona", "hidroclorotiazida", "la insulina glargina", "el inhalador"]
S["persona"] += ["mi tio", "mi suegra", "mi nieto", "mi hermana", "mi abuelo", "mi sobrina", "mi tia"]
S["entidad"] += ["la farmacia", "el hospital de Riohacha", "la EPS subsidiada", "el dispensario"]
S.update({
    "examen": ["la resonancia", "la ecografia", "el TAC", "la endoscopia", "la mamografia", "la colonoscopia",
               "el electrocardiograma", "la radiografia", "los examenes de laboratorio", "la biopsia"],
    "procedimiento": ["la cirugia de vesicula", "la cirugia de cataratas", "las terapias fisicas", "la biopsia",
                      "la cirugia de hernia", "la cesarea programada", "las sesiones de fonoaudiologia"],
    "ciudad": ["Riohacha", "Valledupar", "Barranquilla", "Santa Marta", "Maicao", "Bogota"],
    "veces": ["tres veces", "dos veces", "cuatro veces", "varias veces", "cinco veces"],
    "plata": ["20 mil pesos", "50 mil", "35.000 pesos", "120 mil", "15 mil pesos", "80 mil"],
    "funcionario": ["la recepcionista", "el vigilante", "el medico", "la enfermera", "la de facturacion",
                    "el portero", "la jefe de enfermeria", "el de la ventanilla", "la del call center"],
    "dias": ["un mes", "dos meses", "45 dias", "tres semanas", "mas de un mes", "casi tres meses"],
})

# Piezas de contexto y cierre por tipo (vacías = sin pieza, para que no todos los mensajes las lleven)
PRE = {
    "Peticion": ["", "", "", "una pregunta, ", "quisiera saber, ", "me pueden ayudar, ", "les escribo porque ",
                 "disculpe la molestia, ", "necesito ayuda: ", "buenas, una consulta: "],
    "Queja": ["", "", "", "quiero poner una queja: ", "es increible que ", "estoy muy molesto, ",
              "no es justo, ", "les cuento que ", "otra vez lo mismo, ", "que falta de respeto, "],
    "Reclamo": ["", "", "", "quiero hacer un reclamo, ", "ya van {veces} que ", "otra vez, ", "no puede ser que ",
                "estoy cansada de esto, ", "les escribo para reclamar porque ", "ya no se que hacer, "],
    "Sugerencia": ["", "", "", "tengo una idea: ", "les dejo una sugerencia, ", "con todo respeto, ",
                   "pienso que ", "seria bueno decirles que ", "como usuaria opino que "],
    "Otro": [""],
}
POS = {
    "Peticion": ["", "", "", ", que debo hacer?", ", gracias", ", quedo atento", ", me pueden orientar?",
                 ", es urgente", ", por favor me ayudan", ", a quien me dirijo?"],
    "Queja": ["", "", "", ", eso no puede seguir asi", ", espero que hagan algo", ", que verguenza",
              ", nadie dice nada", ", exijo respeto", ", ya me canse"],
    "Reclamo": ["", "", "", ", exijo una solucion", ", necesito respuesta ya", ", ya no se a quien acudir",
                ", quiero que me respondan", ", eso es un derecho", ", voy a poner tutela"],
    "Sugerencia": ["", "", "", ", ojala lo tengan en cuenta", ", seria de gran ayuda", ", gracias por leer",
                   ", asi mejora el servicio para todos"],
    "Otro": [""],
}

# Núcleos por (tipo, intención). Llaves {slot} se rellenan al azar.
N = {
 ("Peticion", "solicitar_cita"): [
    "necesito cita con {servicio}", "para cuando hay agenda de {servicio} en {lugar}",
    "quiero sacar turno de {servicio} para {persona}", "como pido una cita sin ir hasta {entidad}",
    "hay cupo esta semana con {servicio}", "me mandaron control con {servicio} y no se como pedirlo",
    "{persona} necesita cita de control con {servicio}", "por este medio puedo agendar {servicio}?",
    "me toca la cita de control de {servicio}, donde la pido", "quiero reservar una consulta de {servicio} en la manana",
    "se puede pedir cita para {persona} y para mi el mismo dia", "necesito que me den una cita lo antes posible"],
 ("Peticion", "solicitar_autorizacion"): [
    "necesito que me autoricen {examen}", "me ordenaron {examen} y no se como autorizarlo",
    "donde radico la orden de {examen} para que la aprueben", "pido autorizacion para {procedimiento} de {persona}",
    "el doctor formulo {medicamento} y piden autorizacion, como hago", "que papeles necesito para autorizar {examen}",
    "quiero tramitar la autorizacion de {servicio}", "me pueden autorizar {examen} por aqui",
    "tengo la orden medica de {examen}, a donde la mando"],
 ("Peticion", "solicitar_documento"): [
    "necesito copia de {documento}", "como saco {documento} de {persona}",
    "me pueden mandar {documento} al correo", "requiero {documento} para un tramite",
    "donde reclamo {documento}", "quiero pedir {documento} de la ultima consulta",
    "me piden {documento} en el trabajo, como la consigo", "necesito {documento} firmada por el medico"],
 ("Peticion", "solicitar_interprete"): [
    "{persona} solo habla wayuunaiki, hay interprete?", "necesitamos traductor wayuu para la consulta de {servicio}",
    "pueden poner alguien que hable wayuunaiki en {entidad}", "{persona} no entiende al medico, pido un interprete",
    "hay personal que traduzca al wayuunaiki en {lugar}", "para la cita de {persona} necesito traductor"],
 ("Peticion", "consultar_estado"): [
    "como va mi solicitud de {tiempo}", "en que va mi tramite", "ya revisaron lo que radique {tiempo}?",
    "quisiera saber el estado de mi peticion", "me dieron un radicado, ya hay respuesta?",
    "hace rato mande una solicitud y quiero saber en que quedo", "ya hay respuesta de lo que pedi para {persona}?"],
 ("Peticion", "solicitar_remision"): [
    "necesito que me remitan a {servicio} en {ciudad}", "el medico dijo que {persona} necesita remision a especialista",
    "como hago para que me manden a {ciudad} para {servicio}", "pido la remision a un hospital de mas nivel",
    "aqui en {lugar} no hay {servicio}, necesito remision", "me pueden remitir con el especialista de {servicio}",
    "quiero saber como sacan la remision para {ciudad}", "{persona} necesita ir a {servicio} en {ciudad}, como la remiten"],
 ("Peticion", "afiliacion_eps"): [
    "quiero afiliarme a la EPS", "como afilio a {persona} recien nacido", "necesito cambiarme de EPS",
    "me vine a vivir a {lugar}, como hago la portabilidad", "como incluyo a {persona} como beneficiario",
    "estoy en el regimen subsidiado y quiero pasarme", "me salio que no estoy afiliado, como me afilio",
    "quiero trasladar a mi familia a otra EPS", "que documentos piden para la afiliacion de {persona}"],
 ("Peticion", "solicitar_transporte"): [
    "no tengo como llegar de {lugar} a la cita, dan transporte?", "necesito transporte para {persona} a {ciudad}",
    "la EPS paga el pasaje para ir a {servicio} en {ciudad}?", "como pido los viaticos para el viaje a {ciudad}",
    "vivimos en la rancheria y no hay carro para ir al hospital", "{persona} tiene dialisis en {ciudad}, dan transporte?",
    "pido apoyo con transporte para la cita de {servicio}", "hay ambulancia o carro para llevar pacientes desde {lugar}?"],
 ("Queja", "mal_trato"): [
    "{funcionario} me trato muy mal", "{funcionario} le grito a {persona}", "me humillaron en {entidad}",
    "nos trataron mal por ser wayuu", "{funcionario} fue grosero y no me quiso escuchar",
    "en {entidad} se burlaron de {persona} porque no sabe leer", "{funcionario} nos hablo de mala manera",
    "me ignoraron y me trataron como si no valiera nada", "el trato de {funcionario} fue humillante"],
 ("Queja", "demora_atencion"): [
    "llevo horas esperando en {entidad}", "la fila de {entidad} no avanza", "desde las 5 de la manana estamos aqui y nada",
    "{persona} lleva todo el dia esperando que la atiendan", "en {entidad} de {lugar} atienden demasiado lento",
    "solo hay un medico para toda la gente", "esperamos mas de 6 horas para una consulta",
    "llegamos de {lugar} temprano y todavia no nos llaman", "el call center nunca contesta, llevo horas"],
 ("Queja", "infraestructura"): [
    "en {entidad} no hay agua", "los banos de {entidad} estan sucios", "la sala de espera no tiene sillas",
    "el puesto de salud de {lugar} se esta cayendo", "no hay aire y la gente se desmaya del calor en la fila",
    "en {entidad} se va la luz a cada rato", "el techo del puesto de salud de {lugar} se gotea",
    "no hay camillas suficientes en {entidad}", "el lugar esta lleno de basura"],
 ("Reclamo", "medicamento_no_entregado"): [
    "no me han entregado {medicamento}", "en la farmacia dicen que no hay {medicamento}",
    "llevo {duracion} sin {medicamento}", "me entregaron incompleta la formula de {medicamento}",
    "{persona} se quedo sin {medicamento} porque no lo entregan", "fui {veces} por {medicamento} y nada",
    "me dicen que vuelva la otra semana por {medicamento}", "la formula de {medicamento} se vencio esperando"],
 ("Reclamo", "cita_incumplida"): [
    "me cancelaron la cita de {servicio}", "el especialista de {servicio} no llego", "llegue a la cita y no estaba agendada",
    "me cambiaron la cita sin avisarme", "viajamos desde {lugar} y la cita se cancelo",
    "{persona} tenia cita con {servicio} y no la atendieron", "me reprogramaron la cita {veces}",
    "dijeron que el sistema borro mi cita de {servicio}"],
 ("Reclamo", "cobro_indebido"): [
    "me cobraron {plata} por {servicio}", "me hicieron pagar {medicamento} que es gratis",
    "me cobraron copago siendo del sisben", "en {entidad} me pidieron {plata} para atender a {persona}",
    "me cobraron dos veces la cuota moderadora", "me cobraron {examen} que lo cubre la EPS",
    "me exigieron pagar {plata} por urgencias"],
 ("Reclamo", "servicio_negado"): [
    "me negaron {examen}", "no quisieron atender a {persona}", "la EPS rechazo la orden de {servicio}",
    "dicen que {procedimiento} no lo cubren", "en {entidad} no me atendieron porque no tenia el carnet",
    "me dijeron que no hay convenio y no me atienden", "rechazaron la autorizacion de {medicamento}",
    "le negaron la atencion a {persona} por no estar en el sistema"],
 ("Reclamo", "demora_autorizacion"): [
    "la autorizacion de {examen} lleva {dias} sin salir", "radique la orden hace {dias} y no la autorizan",
    "siempre dicen que la autorizacion esta en tramite", "llevo {dias} esperando que aprueben {procedimiento}",
    "la orden de {servicio} sigue pendiente de autorizacion", "no me han dado respuesta de la autorizacion de {persona}",
    "la autorizacion de {medicamento} no sale y se vence la formula"],
 ("Reclamo", "procedimiento_no_realizado"): [
    "no me han hecho {procedimiento}", "me programaron {procedimiento} y lo cancelaron",
    "los resultados de {examen} no aparecen", "{persona} lleva {dias} esperando {procedimiento}",
    "me hicieron {examen} y nunca entregaron el resultado", "suspendieron {procedimiento} sin explicacion",
    "me dejaron preparado para {examen} y no lo hicieron"],
 ("Sugerencia", "mejorar_atencion"): [
    "deberian dar las citas por whatsapp", "seria bueno que abran mas temprano", "pongan mas medicos en {entidad}",
    "que haya una fila preferencial para {persona} y los abuelos", "que avisen por mensaje cuando cancelen una cita",
    "deberian entregar los medicamentos a domicilio", "que pongan sillas y agua en la sala de espera",
    "que capaciten al personal en buen trato", "que el call center tenga mas lineas",
    "deberian hacer jornadas de salud en {lugar}"],
 ("Sugerencia", "enfoque_intercultural"): [
    "deberian contratar personal wayuu en {entidad}", "que los avisos esten tambien en wayuunaiki",
    "que respeten la medicina tradicional wayuu", "que vayan a las rancherias de {lugar} con brigadas",
    "que los medicos aprendan saludos en wayuunaiki", "que haya un enlace indigena en {entidad}",
    "que expliquen los tratamientos con dibujos para quienes no leen", "que tengan en cuenta a las autoridades tradicionales"],
 ("Otro", "saludo"): ["hola", "buenas", "buenos dias", "buenas tardes", "buenas noches", "hola buen dia",
                      "holaa", "buenas, hay alguien?", "hola que tal", "saludos"],
 ("Otro", "agradecimiento"): ["gracias", "muchas gracias", "ok gracias", "listo gracias", "mil gracias", "dios le pague",
                              "gracias por la informacion", "perfecto, gracias", "muy amable", "bueno gracias, chao"],
 ("Otro", "felicitacion"): [
    "quiero felicitar a {funcionario} por su atencion", "la atencion en {entidad} fue excelente",
    "gracias a {funcionario} que atendio muy bien a {persona}", "felicito al personal de {servicio}",
    "me atendieron rapido y con amabilidad", "el medico de {servicio} fue muy humano con {persona}",
    "quiero reconocer el buen trabajo de {entidad} de {lugar}"],
 ("Otro", "fuera_de_dominio"): [
    "a que hora juega el junior", "me vende una recarga", "necesito trabajo", "cuanto vale el pasaje a {ciudad}",
    "quien gano la loteria", "donde venden mochilas wayuu", "como esta el dolar hoy", "se va el agua manana?",
    "me presta plata", "que dia es el festivo", "hay subsidio de vivienda?", "como saco el pasaporte"],
}

# Frases libres escritas completas (cada una es su propia plantilla).
L = {
 "solicitar_cita": ["mi hija tiene el control de crecimiento y desarrollo vencido, cuando me la pueden ver",
    "buenas, el medico general me dijo que volviera en un mes, ya paso, como pido la cita",
    "soy de Uribia y quiero saber si la odontologa viene esta semana para pedir turno",
    "ando con la presion alta, quiero que me vea un medico pronto"],
 "solicitar_autorizacion": ["el ortopedista dijo que me tienen que operar la rodilla pero primero autorizan, como empiezo",
    "tengo la orden de la resonancia de columna, me dicen que la EPS la tiene que aprobar"],
 "solicitar_documento": ["me accidente en el trabajo el mes pasado y necesito la epicrisis para la ARL",
    "el colegio me pide el certificado de vacunas de la nina, donde lo saco"],
 "solicitar_interprete": ["mi abuela no habla castellano y ayer no entendio nada de lo que le dijo el doctor",
    "la proxima vez que vayamos queremos que este alguien que le explique en su lengua a mi tio"],
 "consultar_estado": ["hace 20 dias deje un papel en la ventanilla y no me han llamado, que paso con eso",
    "PQRS de la semana pasada, todavia nada?"],
 "solicitar_remision": ["el pediatra dice que el nino tiene que verlo un cardiologo pero aqui no hay, ayudenme con eso",
    "en Manaure no hay neurologo y mi esposa tiene migrana fuerte hace meses, la pueden mandar a otra ciudad"],
 "afiliacion_eps": ["tuve a mi bebe hace 15 dias y todavia no aparece en la EPS",
    "trabajaba con contrato y me quede sin empleo, que pasa con mi salud ahora"],
 "solicitar_transporte": ["de la ranchia al hospital son tres horas en burro, no hay forma de llevar a mi mama a las terapias",
    "el tratamiento es en Barranquilla cada 15 dias y no tenemos plata para los pasajes"],
 "mal_trato": ["la doctora ni me miro, escribio todo en el computador y me saco en dos minutos",
    "cuando dije que no entendia el vigilante se rio y me dijo india bruta"],
 "demora_atencion": ["llegue a las 4 de la madrugada a hacer fila para coger ficha y a las 11 todavia nada",
    "mi papa con 80 anos sentado en el piso esperando desde por la manana"],
 "infraestructura": ["el puesto de salud de la comunidad lleva un ano cerrado y con la puerta oxidada",
    "la nevera de las vacunas esta danada y dicen que por eso no vacunan"],
 "medicamento_no_entregado": ["mi mama es hipertensa y lleva 20 dias sin la pastilla de la presion porque en el dispensario no hay",
    "me dieron 10 pastillas de 30 y me dijeron que el resto despues, ya va un mes"],
 "cita_incumplida": ["salimos a las 3 de la manana de la Alta Guajira para la cita y nos dijeron que el especialista no vino",
    "me llamaron un dia antes a decirme que la cita se corria para dentro de dos meses"],
 "cobro_indebido": ["soy indigena y no debo pagar copago pero igual me cobraron en la caja",
    "me hicieron comprar los guantes y las jeringas para atender a mi esposa"],
 "servicio_negado": ["en urgencias no recibieron a mi hijo porque dicen que la EPS no tiene contrato con ellos",
    "me dijeron que esa operacion no esta en el plan y que me toca pagarla particular"],
 "demora_autorizacion": ["ya llevo tres viajes a Riohacha preguntando por la autorizacion y siempre me dicen que todavia no",
    "la orden de la quimioterapia lleva semanas en autorizacion y la enfermedad no espera"],
 "procedimiento_no_realizado": ["me sacaron la muestra hace dos meses y el resultado de la biopsia no aparece",
    "nos dijeron que la cirugia era el lunes, ayune todo el dia y la suspendieron sin explicar"],
 "mejorar_atencion": ["seria bueno que manden recordatorio de la cita un dia antes por whatsapp",
    "propongo que los sabados atiendan aunque sea medio dia para los que trabajamos"],
 "enfoque_intercultural": ["seria bonito que el hospital tenga una enfermera wayuu que hable con las abuelas",
    "las charlas de salud deberian hacerse en la comunidad y en nuestra lengua"],
 "felicitacion": ["la enfermera de la noche fue un angel con mi papa, que Dios la bendiga",
    "por fin una EPS que responde rapido, me solucionaron en dos dias"],
 "fuera_de_dominio": ["alguien sabe si ya pagaron el subsidio del adulto mayor", "manden el numero de la registraduria"],
 "saludo": ["jamaya, hay alguien que me atienda?"],
 "agradecimiento": ["ya me llego el medicamento, muchas gracias por la ayuda"],
}

# Palabras españolas que los usuarios wayuu suelen decir en wayuunaiki (verificadas en el corpus paralelo)
ESP2WAY = {"hijo": ["achonni", "tachon"], "hija": ["tachon"], "mama": ["eii"], "madre": ["eii"], "papa": ["ashi"],
           "medico": ["tottoolu"], "doctor": ["tottoolu"], "enfermo": ["ayuulii"], "enferma": ["ayuulii"],
           "fiebre": ["tuttaa"], "dolor": ["jali"], "cabeza": ["ekii"], "medicina": ["epi"], "medicamento": ["epi"],
           "embarazada": ["ipooluu"], "abuela": ["oushuu"], "abuelo": ["atuushi"], "dinero": ["nneerü"],
           "plata": ["nneerü"], "pagar": ["awalaajaa"], "diarrea": ["e'iijaa"], "gripa": ["shunuin"],
           "sangre": ["asha"], "herida": ["alio'u"], "hermano": ["awala"], "esposa": ["eerüin"], "agua": ["wüin"],
           "comida": ["eküülü"], "hambre": ["jamü"], "cansado": ["mapüsaa"], "enfermedad": ["wanülü"],
           "nino": ["jintüi"], "niño": ["jintüi"], "bebe": ["jintüi"]}
SALUDOS_WAY = ["jamaya", "jamayaa", "wattachonya", "jamaya jia"]
CIERRES_WAY = ["anayaawatsüja'a", "analu'ulu pümüin"]


def wayuunizar(txt, rng):
    """Cambia 1-3 palabras por su forma en wayuunaiki y, a veces, añade saludo/cierre wayuu."""
    toks, idx = txt.split(), []
    for i, t in enumerate(toks):
        if re.sub(r"[^\wñ]", "", gd_quitar(t.lower())) in ESP2WAY:
            idx.append(i)
    rng.shuffle(idx)
    for i in idx[:rng.randint(1, 3)]:
        limpio = re.sub(r"[^\wñ]", "", gd_quitar(toks[i].lower()))
        toks[i] = rng.choice(ESP2WAY[limpio]) + re.sub(r"^[\wñáéíóúü]+", "", toks[i])
    out = " ".join(toks)
    if not idx or rng.random() < .35:
        out = rng.choice(SALUDOS_WAY) + " " + out
    if rng.random() < .15:
        out += " " + rng.choice(CIERRES_WAY)
    return out


def gd_quitar(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.replace("ñ", "\0")) if not unicodedata.combining(c)).replace("\0", "ñ")


CAMBIOS = [("mucho", "musho"), ("hace", "ase"), ("necesito", "nesesito"), ("medico", "dotor"), ("cita", "sita"),
           ("llevo", "yevo"), ("quiero", "kiero"), ("que", "q"), ("porque", "xq"), ("por favor", "xfa")]


def _render(tpl, rng):
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(S[m.group(1)]), tpl)


def _frase_base(tipo, nucleo, rng):
    if tipo == "Otro":
        return _render(nucleo, rng)
    base = _render(rng.choice(PRE[tipo]) + nucleo + rng.choice(POS[tipo]), rng)
    return re.sub(r"\b(\w+) \1\b", r"\1", base)          # "opino que que..." -> "opino que..."


def _variantes(base, tipo, rng):
    """3 variantes de una frase base usando los transformadores del generador original."""
    gd.rng = rng
    simp = gd.a_simplificado(base)
    for a, b in CAMBIOS:                                    # más errores típicos de escritura
        if rng.random() < .25:
            simp = re.sub(rf"\b{a}\b", b, simp)
    return [("formal", gd.a_formal(base, tipo)), ("coloquial", gd.a_coloquial(base)), ("simplificado", simp)]


def generar_v2(por_intencion=115, seed=42):
    rng = random.Random(seed)
    filas, vistos = [], set()
    externo = {normalizar(t) for t in pd.read_csv(TEST_EXTERNO, keep_default_na=False).texto}
    for (tipo, intencion), nucleos in N.items():
        bases = [(f"v2_{intencion}_c{j}", n, False) for j, n in enumerate(nucleos)]
        bases += [(f"v2_{intencion}_l{k}", t, True) for k, t in enumerate(L.get(intencion, []))]
        # Sugerencia es la clase más débil: se refuerza. Saludos/agradecimientos/fuera de dominio son cortos.
        n_obj = {"Sugerencia": int(por_intencion * 1.8)}.get(tipo, por_intencion)
        if tipo == "Otro" and intencion != "felicitacion":
            n_obj = por_intencion // 2
        intentos, hechas = 0, 0
        libres = [b for b in bases if b[2]]
        for pid, texto, _ in libres:                         # las libres siempre entran una vez
            hechas += _agregar(filas, vistos, externo, pid, texto, tipo, intencion, rng)
        while hechas < n_obj and intentos < n_obj * 20:
            intentos += 1
            pid, nucleo, _ = rng.choice([b for b in bases if not b[2]])
            hechas += _agregar(filas, vistos, externo, pid, _frase_base(tipo, nucleo, rng), tipo, intencion, rng)
    d = pd.DataFrame(filas)
    d.insert(0, "id_msg", [f"N{i:05d}" for i in range(len(d))])
    return d


def _agregar(filas, vistos, externo, pid, base, tipo, intencion, rng):
    clave = normalizar(base)
    if clave in vistos or clave in externo:
        return 0
    vistos.add(clave)
    for registro, txt in _variantes(base, tipo, rng):
        way = rng.random() < (.30 if registro == "simplificado" else .12)
        if way:
            txt = wayuunizar(txt, rng)
        if normalizar(txt) in externo:
            continue
        filas.append(dict(texto=txt, tipo_pqrs=tipo, intencion=intencion, registro=registro, wayuunaiki=way,
                          plantilla_id=pid, fuente="sintetico_combinatorio_v2"))
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--por-intencion", type=int, default=115, help="frases base por intención (×3 variantes)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    original = gd.generar()
    nuevo = generar_v2(a.por_intencion, a.seed)
    total = pd.concat([original, nuevo], ignore_index=True)
    total.to_csv(SALIDA, index=False)
    print(f"original {len(original):,} + nuevos {len(nuevo):,} = {len(total):,} filas -> {SALIDA}")
    print(f"frases base nuevas: {nuevo.plantilla_id.nunique()} plantillas, {len(nuevo)//3:,} frases base")
    print(total.groupby("tipo_pqrs").size().to_string())


if __name__ == "__main__":
    main()
