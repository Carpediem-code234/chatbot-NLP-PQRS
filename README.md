# Chatbot incluyente de PQRS en salud por WhatsApp

Chatbot de procesamiento de lenguaje natural (NLP) que recibe **Peticiones, Quejas, Reclamos y Sugerencias (PQRS)** de atención en salud por WhatsApp. Está pensado para poblaciones vulnerables de La Guajira: entiende mensajes en **lenguaje simplificado** (personas con baja alfabetización, errores ortográficos, abreviaturas) y reconoce palabras básicas en **wayuunaiki**.

> ⚠️ Proyecto académico. Los modelos se entrenaron con **datos sintéticos** y sus métricas **no representan** el desempeño con usuarios reales. El vocabulario en wayuunaiki está **pendiente de validación por hablantes nativos**.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 · Datos y EDA | Fuentes de datos, EDA del corpus español–wayuunaiki, cobertura de salud, glosario, dataset PQRS sintético y su EDA | ✅ `notebooks/01_…` |
| 2 · Desarrollo y modelado | Preprocesamiento, partición sin fuga, baseline + modelo propuesto, ajuste de hiperparámetros, evaluación | ✅ `notebooks/02_…`, `src/` |
| 2 · BETO (BERT en español) | Código listo; requiere GPU | ⏳ ejecutar en Colab |
| 3 · Integración y despliegue | Bot en WhatsApp Business (Cloud API), API documentada, radicación de PQRS | ✅ `app/`, [`docs/DESPLIEGUE_WHATSAPP.md`](docs/DESPLIEGUE_WHATSAPP.md) |

## Cómo funciona

El usuario escribe al número de WhatsApp; Meta reenvía el mensaje a nuestra API (`POST /webhook`), que decide y responde:

```
mensaje de WhatsApp
   │
   ├─ R0  ¿sin palabras (vacío, solo emojis)?  ──► "No entendí, ¿me cuentas qué necesitas?"
   ├─ R1  ¿alerta clínica (convulsiona, no respira, dolor de pecho, intoxicación,
   │       sangrado, quemaduras, riesgo suicida…)? ──► remitir a urgencias (123)
   │
   ├─ normalización: minúsculas, sin tildes, abreviaturas (q, xq, pa), ortografía (sita→cita),
   │                 wayuunaiki→español con glosario (tottoolu→médico, ayuulii→enfermo)
   ├─ modelo M3: TF-IDF palabras ∪ n-gramas de caracteres ∪ rasgos estructurales → SVM calibrada
   │
   └─ R2  ¿confianza < τ (0,55)? ──► preguntar con botones: pedir / queja / reclamo / sugerencia
          si no ──► "Entendí que es un reclamo sobre un medicamento que no te entregaron. ¿Es correcto?"
                         │ sí
                         ├─ pide el documento (o "no tengo")
                         └─ radica en SQLite ──► "Tu número de radicado es PQRS-20261119-00001"
```

Conversación de ejemplo:

```
👤 no me han entregado la insulina de mi mama hace 15 dias
🤖 Entendí que es un *reclamo* (un servicio que no te dieron o te dieron mal),
   sobre *un medicamento que no te entregaron*. ¿Es correcto?  1 Sí ✅  2 No ❌
👤 si
🤖 Para registrar tu solicitud necesito tu número de documento. Si no lo tienes, escribe *no tengo*.
👤 1.234.567.890
🤖 ✅ Listo. Registramos tu reclamo. Tu número de radicado es: *PQRS-20261119-00001*
   Te responderemos en máximo 15 días hábiles.
```

Además: las reglas de urgencia y de "hablar con un agente" funcionan en **cualquier** momento; "aa" (sí en wayuunaiki) sirve para confirmar; `cancelar` reinicia; las notas de voz reciben un aviso; el usuario puede preguntar por su radicado ("como va el PQRS-…"); y un funcionario responde las PQRS desde la API y el usuario recibe el aviso por WhatsApp.

## Resultados

Dataset actual: **10.731 mensajes** sintéticos de **310 plantillas**, 5 tipos de PQRS y 23 intenciones (más 30 mensajes sin contenido para la regla R0). La partición es **por plantilla** (20 % test, 20 % validación de cada intención), de modo que validación y test solo tienen **plantillas nunca vistas en entrenamiento**; el test externo son 68 mensajes redactados a mano.

F1-macro en la clasificación del **tipo de PQRS** (5 clases):

| Modelo | F1 val | AUC val | F1 test (10 particiones) | F1 externo (10 particiones) |
|---|---|---|---|---|
| M0 · Clase mayoritaria | 0,093 | 0,500 | — | — |
| M1 · Baseline: TF-IDF palabras + regresión logística | **0,893** | 0,990 | **0,894 ± 0,048** | 0,810 ± 0,046 |
| M2 · TF-IDF + Naive Bayes complementario | 0,813 | 0,981 | 0,833 ± 0,033 | 0,778 ± 0,029 |
| **M3 · Propuesto** (desplegado) | 0,809 | **0,993** | 0,878 ± 0,046 | **0,863 ± 0,040** |

- Con más datos, el **baseline M1 alcanza a M3** en plantillas sintéticas no vistas (la diferencia está dentro de la variación entre particiones).
- M3 sigue siendo el mejor en el **test externo** redactado a mano (+0,05 sobre M1), que es lo más parecido a usuarios reales; por eso es el modelo del bot. Su punto débil es la clase *Otro* (saludos, fuera de dominio), que confunde con peticiones.
- El umbral τ se elige en validación como el menor que da ≥ 90 % de exactitud en lo que el bot decide solo: **τ = 0,55**, con el que decide solo en ~95 % de los mensajes de validación (exactitud ~91 %); el resto lo resuelve preguntando. Curva completa en `results/umbral_tau.csv`.
- La **intención fina** (23 clases) mejoró pero sigue siendo débil: F1 ≈ 0,58 en test y 0,55 en el externo con M3. El bot la usa solo para redactar la confirmación, que el usuario siempre valida.
- Las métricas en `notebooks/02_modelado_PQRS.ipynb` corresponden al dataset anterior (≈ 3.100 mensajes); los valores vigentes son los de `results/`.

Bitácora de experimentos: `results/experimentos.csv` · métricas: `results/resultados_modelos.csv` · figuras: `results/figuras/`.

## Estructura

```text
data/
  raw/                pqrs_salud_sintetico.csv, pqrs_test_externo.csv, glosario_salud_esp_way.csv
  external/           corpus wayuu.zip (no versionado; ver data/README.md)
notebooks/
  01_EDA_corpus_wayuu_y_PQRS.ipynb    Fase 1: fuentes, EDA, calidad de datos, glosario, dataset PQRS
  02_modelado_PQRS.ipynb              Fase 2: preprocesamiento, modelos, ajuste, evaluación, BETO
app/                  Fase 3: chatbot de WhatsApp
  main.py             API FastAPI: webhook de WhatsApp, /api/clasificar, /api/pqrs (admin), /health
  dialogo.py          conversación: clasificar → confirmar / menú → documento → radicar; consultar radicado
  seguridad.py        reglas de urgencia y de agente (basadas en las de la rama Whatsapp-Api-integracion)
  mensajes.py         textos en lenguaje sencillo
  almacen.py          SQLite: PQRS, estado de conversaciones, eventos, mensajes procesados
  whatsapp.py         cliente de la Cloud API y verificación de firma de Meta
  simular.py          conversación de prueba en la terminal, sin Meta
src/
  config.py           rutas, clases, hiperparámetros elegidos, umbral τ
  generar_dataset.py  generador reproducible del dataset sintético
  preprocesamiento.py normalización, glosario wayuunaiki, rasgos estructurales
  datos.py            limpieza, deduplicación, partición por plantilla sin fuga
  modelos.py          M0, M1, M2, M3
  evaluacion.py       métricas, 10 particiones, curva del umbral τ
  bot.py              clasificador con reglas R0/R1/R2 (Fase 2)
  entrenar.py         entrena, evalúa, elige τ y guarda models/modelo_pqrs_m3.joblib
  predecir.py         clasifica mensajes desde la terminal
tests/
  test_preprocesamiento.py, test_datos.py, test_modelos.py   Fase 2
  test_bot_whatsapp.py  reglas de seguridad, diálogo, webhook y API
results/              métricas, bitácora y figuras
models/               modelo entrenado (no versionado; se regenera)
docs/
  DESPLIEGUE_WHATSAPP.md   conectar el bot a un número de WhatsApp
  PLAN_DE_COMMITS_FASE3.md orden de commits de esta fase
Dockerfile            imagen del servicio (entrena el modelo al construirse)
.env.example          variables de entorno necesarias (copiar a .env)
```

## Instalación y uso

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.entrenar             # entrena, evalúa y guarda el modelo (≈ 2 min en CPU; --rapido omite las 10 particiones)
python -m pytest                   # 97 pruebas (las del bot se omiten si no hay modelo entrenado)
python -m app.simular              # conversar con el bot en la terminal, sin WhatsApp
```

### Chatbot en WhatsApp

```bash
cp .env.example .env               # completar credenciales de Meta
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API documentada en `http://localhost:8000/docs`. Para conectarlo a un número de WhatsApp real (app de Meta, token, URL pública con ngrok/cloudflared o Docker, webhook) sigue **[docs/DESPLIEGUE_WHATSAPP.md](docs/DESPLIEGUE_WHATSAPP.md)**.

| Endpoint | Uso |
|---|---|
| `GET /webhook` | verificación del webhook por Meta |
| `POST /webhook` | mensajes entrantes de WhatsApp (firma `X-Hub-Signature-256` verificada) |
| `POST /api/clasificar` | clasificar un texto (prueba del modelo, no radica) |
| `GET /api/pqrs`, `GET /api/pqrs/{radicado}`, `GET /api/pqrs/resumen` | consultar PQRS (cabecera `X-Admin-Token`) |
| `PATCH /api/pqrs/{radicado}` | cambiar estado / responder; avisa al usuario por WhatsApp |
| `GET /api/eventos` | urgencias detectadas, solicitudes de agente, notas de voz |
| `GET /health` | estado del servicio |

`python -m src.generar_dataset` regenera las primeras 3.165 filas del CSV versionado (plantillas, ruido y audio transcrito) con la semilla por defecto; hay una prueba que lo verifica. Las 7.566 filas de la ampliación (`fuente = sintetico_combinatorio_v2`) aún **no tienen script generador** en el repositorio.

Para ejecutar el notebook 01 se necesita el corpus en `data/external/wayuu.zip`. El modelo BETO (notebook 02, §8) se ejecuta en Google Colab con GPU cambiando `RUN_BERT = True`.

## Limitaciones

- Datos sintéticos generados por plantillas: vocabulario reducido y métricas optimistas. La ampliación a 10.731 mensajes es combinatoria (mismas ideas con otras palabras), así que suma menos diversidad de la que sugiere el tamaño.
- La clase *Otro* (saludos, agradecimientos, fuera de dominio) es la más débil: el bot tiende a tratarla como petición.
- El test externo lo redactó el mismo equipo que diseñó las plantillas.
- Glosario wayuunaiki pequeño y sin validar; faltan términos clave (*hospital, cita, vacuna, salud*).
- La regla de urgencias es una lista de frases, no un modelo: no reemplaza el triage.
- El bot no transcribe notas de voz todavía; con el número de prueba de Meta solo atiende a 5 números registrados.
- Fuera de las 24 h desde el último mensaje del usuario, WhatsApp exige plantillas aprobadas para escribirle (no implementado).

## Fuentes de datos

Ver [`data/README.md`](data/README.md): corpus español–wayuunaiki (Universidad de los Andes), vocabulario de enfermedades del Portal UNICISO (Kelly Johana López Epieyu y Sharol Robles Epieyu) y normativa colombiana de PQRS.
