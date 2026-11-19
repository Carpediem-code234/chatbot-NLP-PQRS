# Datos

| Archivo | Fuente | Descripción |
|---|---|---|
| `raw/pqrs_salud_sintetico.csv` | **F2 · Generación sintética propia** (`python -m src.generar_dataset`, semilla 42) | 3.165 mensajes tipo WhatsApp etiquetados con `tipo_pqrs` (5 clases) e `intencion` (17 + `sin_contenido`). Incluye registro formal / coloquial / simplificado, ~10 % con palabras en wayuunaiki y ruido de WhatsApp inyectado a propósito (reenvíos, mensajes vacíos, audios transcritos). |
| `raw/pqrs_test_externo.csv` | Redactado a mano, fuera de las plantillas | 68 mensajes para medir generalización real. Lo redactó el mismo equipo que diseñó las plantillas: idealmente debe reemplazarse por mensajes escritos por otras personas. |
| `raw/glosario_salud_esp_way.csv` | **F3** derivado del corpus (F1) + **F6** vocabulario UNICISO | 36 términos de salud español → wayuunaiki. Columna `validado_hablante_nativo` en `False`: **requiere validación por hablantes nativos** antes de usarse en respuestas. |
| `external/wayuu.zip` | **F1 · Corpus paralelo público español–wayuunaiki** (consolidado por la Universidad de los Andes, AmericasNLP 2024) | 82.218 pares en su versión `completo_v2`. **No se versiona** (49 MB): descargarlo y colocarlo en `data/external/wayuu.zip`. Confirmar licencia antes de redistribuir. |

## Fuentes y créditos

- **F1** — Corpus de traducción español–wayuunaiki, Colombian Languages, Universidad de los Andes: <https://colombialanguages.virtual.uniandes.edu.co/resources/>
- **F6** — Audios "Enfermedades en wayuunaiki", Portal UNICISO, colaboración wayuu de Kelly Johana López Epieyu y Sharol Robles Epieyu: <https://www.portaluniciso.com/categoria-producto/wayuunaiki/>. Solo se usan los 10 términos escritos, con crédito; los audios no se redistribuyen.
- **F4** — Definiciones de petición, queja, reclamo y sugerencia: Ley 1755 de 2015 y lineamientos de la Superintendencia Nacional de Salud.

## Esquema de `pqrs_salud_sintetico.csv`

| Columna | Descripción |
|---|---|
| `id_msg` | identificador único (`_r` = reenvío, `V…` = mensaje sin contenido, `A…` = audio transcrito) |
| `texto` | mensaje del usuario |
| `tipo_pqrs` | `Peticion`, `Queja`, `Reclamo`, `Sugerencia`, `Otro` |
| `intencion` | intención fina (p. ej. `solicitar_cita`, `medicamento_no_entregado`) |
| `registro` | `formal`, `coloquial`, `simplificado` |
| `wayuunaiki` | `True` si mezcla palabras en wayuunaiki |
| `plantilla_id` | plantilla de origen: **se usa para particionar sin fuga** |
| `fuente` | `sintetico_plantillas`, `sintetico_ruido`, `sintetico_audio_transcrito` |
