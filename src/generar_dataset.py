"""Generador reproducible del dataset SINTÉTICO de PQRS en salud (mensajes tipo WhatsApp).

Uso:
    python -m src.generar_dataset                 # escribe data/raw/pqrs_salud_sintetico.csv
    python -m src.generar_dataset --n 5000 --seed 7 --salida otra_ruta.csv

Columnas: id_msg, texto, tipo_pqrs, intencion, registro, wayuunaiki, plantilla_id, fuente.
Las palabras en wayuunaiki provienen del corpus paralelo (Fase 1) y requieren validación
por hablantes nativos.
"""
import argparse
import random
import re
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW

RNG_SEED = 42
rng = random.Random(RNG_SEED)

SLOTS = {
    "servicio": ["medicina general", "odontologia", "pediatria", "control prenatal", "vacunacion",
                 "laboratorio", "ginecologia", "medicina interna", "urgencias", "psicologia", "nutricion"],
    "entidad": ["la EPS", "el hospital", "el puesto de salud", "la IPS", "el centro de salud", "la clinica"],
    "lugar": ["Riohacha", "Uribia", "Maicao", "Manaure", "Albania", "Dibulla", "Nazareth", "la Alta Guajira",
              "la rancheria", "Fonseca"],
    "medicamento": ["acetaminofen", "losartan", "metformina", "insulina", "amoxicilina", "ibuprofeno",
                    "sales de rehidratacion", "acido folico", "salbutamol", "enalapril"],
    "duracion": ["una semana", "15 dias", "un mes", "dos semanas", "tres dias"],
    "tiempo": ["hace una semana", "hace 15 dias", "hace un mes", "desde ayer", "hace tres meses", "desde el lunes",
               "hace dos semanas"],
    "persona": ["mi hijo", "mi hija", "mi mama", "mi papa", "mi esposa", "mi abuela", "mi bebe", "mi esposo"],
    "documento": ["la historia clinica", "un certificado medico", "la incapacidad", "los resultados de laboratorio",
                  "la formula medica", "el carnet de vacunas"],
}

# Plantillas por (tipo, intención). Se escriben en registro "base"; luego se transforman por registro.
T = {
 ("Peticion", "solicitar_cita"): [
    "necesito una cita de {servicio} para {persona}",
    "quiero pedir cita con {servicio} en {lugar}",
    "como hago para sacar una cita de {servicio}",
    "me pueden dar una cita de {servicio} lo mas pronto posible",
    "solicito asignacion de cita de {servicio} para {persona}"],
 ("Peticion", "solicitar_autorizacion"): [
    "necesito la autorizacion para {servicio} que me mando el medico",
    "como tramito la autorizacion de {medicamento} en {entidad}",
    "quiero que me autoricen el examen de {servicio}",
    "solicito autorizacion de la orden de {servicio} para {persona}"],
 ("Peticion", "solicitar_documento"): [
    "necesito {documento} de {persona}",
    "como pido {documento} en {entidad}",
    "solicito copia de {documento}",
    "me pueden enviar {documento} por aqui"],
 ("Peticion", "solicitar_interprete"): [
    "necesito un interprete de wayuunaiki para la cita de {persona}",
    "{persona} no habla bien espanol, hay alguien que traduzca en {entidad}",
    "pido que en la cita de {servicio} haya traductor wayuu"],
 ("Peticion", "consultar_estado"): [
    "quiero saber como va mi solicitud que mande {tiempo}",
    "en que va el tramite que radique {tiempo}",
    "me dieron un numero de radicado y no me han respondido, en que va"],
 ("Queja", "mal_trato"): [
    "en {entidad} de {lugar} me atendieron muy mal, la persona de la ventanilla me grito",
    "el medico de {servicio} fue grosero con {persona}",
    "me trataron mal en {entidad} por no hablar bien espanol",
    "la enfermera no nos quiso atender y nos hablo feo"],
 ("Queja", "demora_atencion"): [
    "llevo horas esperando en {entidad} de {lugar} y nadie me atiende",
    "{persona} esta en urgencias desde ayer y todavia no hay medico que atienda",
    "la fila en {entidad} es demasiado larga, esperamos todo el dia",
    "llegamos temprano de {lugar} y nos atendieron en la noche"],
 ("Queja", "infraestructura"): [
    "el puesto de salud de {lugar} no tiene agua ni luz",
    "en {entidad} no hay sillas y la gente espera parada al sol",
    "los banos de {entidad} estan danados y sucios",
    "no hay ambulancia en {lugar} para traer a los enfermos"],
 ("Reclamo", "medicamento_no_entregado"): [
    "no me han entregado el {medicamento} que me formularon {tiempo}",
    "fui a la farmacia y no habia {medicamento} para {persona}",
    "me dieron la formula de {medicamento} pero la farmacia dice que no hay",
    "llevo {duracion} sin {medicamento} y lo necesito para {persona}"],
 ("Reclamo", "cita_incumplida"): [
    "me cancelaron la cita de {servicio} y no me la reprogramaron",
    "viajamos desde {lugar} para la cita y el medico no llego",
    "tenia cita de {servicio} {tiempo} y nunca me llamaron",
    "me cambiaron la cita de {servicio} sin avisar"],
 ("Reclamo", "cobro_indebido"): [
    "me cobraron por {servicio} y eso lo cubre {entidad}",
    "me hicieron pagar una cuota que no me corresponde",
    "en {entidad} me cobraron el {medicamento} que es gratis",
    "me exigieron pagar para atender a {persona} en urgencias"],
 ("Reclamo", "servicio_negado"): [
    "me negaron la autorizacion de {servicio} sin explicacion",
    "no quisieron atender a {persona} porque dicen que no esta afiliado",
    "{entidad} se niega a dar la orden de {servicio}",
    "me dijeron que no me pueden atender en {lugar}"],
 ("Sugerencia", "mejorar_atencion"): [
    "seria bueno que en {entidad} hubiera mas personal en las mananas",
    "sugiero que den turnos por mensaje para no hacer fila",
    "deberian poner un toldo para la gente que espera afuera en {entidad}",
    "recomiendo que el personal explique con palabras sencillas"],
 ("Sugerencia", "enfoque_intercultural"): [
    "deberian tener personal que hable wayuunaiki en {entidad}",
    "seria bueno que hagan brigadas de salud en {lugar}",
    "sugiero que los avisos esten en wayuunaiki y en espanol",
    "propongo que respeten la medicina tradicional wayuu en la atencion"],
 ("Otro", "saludo"): ["hola", "buenos dias", "buenas tardes", "hola buenas", "buenas noches, una pregunta"],
 ("Otro", "agradecimiento"): ["gracias", "muchas gracias por la ayuda", "ok gracias", "listo, gracias", "muy amable"],
 ("Otro", "fuera_de_dominio"): ["a que hora juega el junior", "quien gano la loteria", "me vende una recarga",
                                "necesito trabajo", "cuanto cuesta el pasaje a Maicao"],
}

# Distribución objetivo (desbalance REALISTA y deliberado: las sugerencias son las menos frecuentes)
TIPO_P = {"Peticion": .32, "Reclamo": .27, "Queja": .21, "Otro": .13, "Sugerencia": .07}
REG_P = {"formal": .20, "coloquial": .50, "simplificado": .30}
N_MENSAJES = 3000

# Palabras en wayuunaiki ATESTIGUADAS en el corpus paralelo (sección 7.1); requieren validación de hablante nativo
WAY_LEX = {"hola": "jamaya", "buenos dias": "wattachonya", "gracias": "anayaawatsüja'a",
           "enfermo": "ayuulii", "medico": "tottoolu", "fiebre": "tuttaa", "si": "aa", "no": "nnojo"}

FORMAL_PRE = ["Buenos días, ", "Cordial saludo. ", "Señores, ", "Por medio de la presente, "]
FORMAL_POS = [". Agradezco su atención.", ". Quedo atento.", ". Muchas gracias.", "."]
COLOQ_PRE = ["buenas ", "hola ", "", "", "porfa ", "oiga "]
COLOQ_POS = ["", "", " porfa", " 🙏", " ayuda", "!!", " ??"]
TILDES = {"medico": "médico", "clinica": "clínica", "autorizacion": "autorización", "rancheria": "ranchería",
          "solicitud": "solicitud", "espanol": "español", "bebe": "bebé", "mama": "mamá", "papa": "papá",
          "dias": "días", "danados": "dañados", "mananas": "mañanas", "historia clinica": "historia clínica",
          "odontologia": "odontología", "pediatria": "pediatría", "vacunacion": "vacunación",
          "ginecologia": "ginecología", "psicologia": "psicología", "nutricion": "nutrición",
          "acetaminofen": "acetaminofén", "losartan": "losartán", "acido": "ácido", "interprete": "intérprete",
          "asignacion": "asignación", "tramite": "trámite", "atencion": "atención", "explicacion": "explicación",
          "afiliado": "afiliado", "examenes": "exámenes", "rehidratacion": "rehidratación", "numero": "número"}
# Errores frecuentes en escritura con baja alfabetización / teclado del celular
SIMPLE_SUBS = [("ci", "si"), ("ce", "se"), ("qu", "k"), ("que", "q"), ("por que", "xq"), ("para", "pa"),
               ("ll", "y"), ("v", "b"), ("h", ""), ("medicamento", "medisina"), ("cita", "sita"),
               ("necesito", "nesesito"), ("hace", "ase")]
ABREV = [("que", "q"), ("por favor", "xfa"), ("para", "pa"), ("tambien", "tb"), ("porque", "xq")]

def render(tpl):
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(SLOTS[m.group(1)]), tpl)

def a_formal(t, tipo=None):
    for k, v in sorted(TILDES.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{k}\b", v, t)
    if tipo == "Otro":                                 # saludos/agradecimientos: solo mayúscula inicial
        return t[0].upper() + t[1:] + "."
    t = rng.choice(FORMAL_PRE) + t
    t = t[0].upper() + t[1:]
    if t.lower().startswith(("como", "en que", "quiero saber")):
        t = "¿" + t + "?"
    return t + ("" if t.endswith("?") else rng.choice(FORMAL_POS))

def a_coloquial(t):
    for k, v in ABREV:
        if rng.random() < .35:
            t = re.sub(rf"\b{k}\b", v, t)
    return rng.choice(COLOQ_PRE) + t + rng.choice(COLOQ_POS)

def a_simplificado(t):
    # estilo telegráfico: se omiten algunos artículos/preposiciones
    t = " ".join(w for w in t.split() if not (w in {"la", "el", "los", "las", "de", "un", "una"} and rng.random() < .4))
    for a, b in SIMPLE_SUBS:
        if rng.random() < .30:
            t = t.replace(a, b)
    return t.upper() if rng.random() < .15 else t      # algunos escriben todo en mayúscula

def add_wayuunaiki(t):
    # code-switching: saludo y/o un término de salud en wayuunaiki
    if rng.random() < .6:
        t = rng.choice([WAY_LEX["hola"], WAY_LEX["buenos dias"]]) + " " + t
    for es in ("enfermo", "medico", "fiebre"):
        t = re.sub(rf"\b{es}\b", WAY_LEX[es], t)
    if rng.random() < .4:
        t += " " + rng.choice(["ayuulii", "tuttaa"])   # 'enfermo', 'fiebre'
    return t

def generar(n=N_MENSAJES, seed=RNG_SEED):
    """Genera el dataset completo (mensajes + ruido de WhatsApp), barajado."""
    global rng, RNG_SEED
    rng, RNG_SEED = random.Random(seed), seed
    rows = []
    tipos, pt = zip(*TIPO_P.items())
    regs, pr = zip(*REG_P.items())
    for i in range(n):
        tipo = rng.choices(tipos, pt)[0]
        intencion = rng.choice([k[1] for k in T if k[0] == tipo])
        reg = rng.choices(regs, pr)[0]
        j = rng.randrange(len(T[(tipo, intencion)]))
        txt = render(T[(tipo, intencion)][j])
        # 8 % de quejas/reclamos con una segunda queja/reclamo en el mismo mensaje (ambigüedad de etiqueta)
        if tipo in ("Queja", "Reclamo") and rng.random() < .08:
            otro = rng.choice([k for k in T if k[0] in ("Queja", "Reclamo") and k[1] != intencion])
            txt += " y ademas " + render(rng.choice(T[otro]))
        txt = a_formal(txt, tipo) if reg == "formal" else (a_coloquial(txt) if reg == "coloquial" else a_simplificado(txt))
        way = rng.random() < (.20 if reg == "simplificado" else .06)
        if way:
            txt = add_wayuunaiki(txt)
        rows.append(dict(id_msg=f"M{i:05d}", texto=txt, tipo_pqrs=tipo, intencion=intencion,
                         registro=reg, wayuunaiki=way, plantilla_id=f"{intencion}_{j}",
                         fuente="sintetico_plantillas"))
    d = pd.DataFrame(rows)

    # ── Ruido realista de WhatsApp (proporciones bajas, documentadas) ────────────
    # (a) reenvíos: el mismo usuario manda dos veces el mismo mensaje
    reenv = d.sample(frac=.04, random_state=RNG_SEED).assign(id_msg=lambda x: x.id_msg + "_r")
    # (b) mensajes vacíos o solo emojis / stickers (sin texto útil)
    vac = pd.DataFrame(dict(id_msg=[f"V{i:04d}" for i in range(30)],
                            texto=[rng.choice(["", " ", "👍", "🙏🙏", "😡", "...", "?"]) for _ in range(30)],
                            tipo_pqrs="Otro", intencion="sin_contenido", registro="coloquial",
                            wayuunaiki=False, plantilla_id="ruido", fuente="sintetico_ruido"))
    # (c) audios transcritos: mensajes muy largos que concatenan varias frases
    largos = []
    for i in range(15):
        k = rng.choice([k for k in T if k[0] in ("Queja", "Reclamo")])
        j = rng.randrange(len(T[k]))
        frases = [render(T[k][j])] + [render(rng.choice(T[k])) for _ in range(rng.randint(7, 13))]
        largos.append(dict(id_msg=f"A{i:04d}", texto=" eee pues ".join(frases), tipo_pqrs=k[0], intencion=k[1],
                           registro="coloquial", wayuunaiki=False, plantilla_id=f"{k[1]}_{j}",
                           fuente="sintetico_audio_transcrito"))
    d = pd.concat([d, reenv, vac, pd.DataFrame(largos)], ignore_index=True)
    return d.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)   # barajar



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_MENSAJES, help="mensajes base a generar")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--salida", type=Path, default=DATA_RAW / "pqrs_salud_sintetico.csv")
    a = ap.parse_args()
    d = generar(a.n, a.seed)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.salida, index=False)
    print(f"{len(d):,} mensajes -> {a.salida}")
    print(d.tipo_pqrs.value_counts().to_string())


if __name__ == "__main__":
    main()
