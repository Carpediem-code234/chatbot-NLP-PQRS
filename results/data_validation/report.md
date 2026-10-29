# Informe de validación de datos (Fase 2, etapa 1)

Filas: 658 | críticas: 0 | advertencias: 7 | ok: 14

| Comprobación | Estado | Resumen |
|---|---|---|
| columnas | ok | {"faltantes": [], "extra": []} |
| tipos_texto | ok | {"columnas_no_texto": []} |
| textos_vacios | ok | {"n": 0} |
| ids_unicos | ok | {"duplicados": 0} |
| etiquetas_validas | ok | {"invalidas": [], "ausentes": [], "n_clases": 16} |
| splits_validos | ok | {"invalidos": [], "conteo": {"train": 398, "test": 133, "dev": 127}} |
| estilos_validos | ok | {"invalidos": []} |
| cobertura_intencion_split | warning | {"sin_ejemplos": [], "menos_de_5": [["no_entendido", "test", 3]], "min_por_celda": 3} |
| duplicados_exactos | warning | {"filas": 2, "ids": ["PQRS-0560", "PQRS-0654"]} |
| duplicados_normalizados | warning | {"grupos": 65, "filas_en_grupos": 155, "filas_redundantes": 90, "textos_unicos": 568, "unicos_por_split": {"dev": 108, "test": 117, "train": 343}, "grupos_que_cruzan_splits": 0, "grupos_con_etiquetas_ |
| conflicto_mismo_texto_distinta_etiqueta | ok | {"n": 0} |
| fuga_template_id | ok | {"plantillas": 240, "plantillas_en_mas_de_un_split": 0, "plantillas_por_split": {"dev": 48, "test": 48, "train": 144}} |
| plantilla_una_sola_intencion | ok | {"plantillas_con_varias_intenciones": 0} |
| split_determinado_por_indice_plantilla | warning | {"descripcion": "cada índice final de template_id cae en una sola partición", "tabla": {"0": {"dev": 0, "test": 0, "train": 139}, "1": {"dev": 0, "test": 0, "train": 122}, "2": {"dev": 0, "test": 0, " |
| solapamiento_dev_test_vs_train | ok | {"dev": {"sim_max_media": 0.492, "sim_max_mediana": 0.463, "n_sim_ge_0.9": 0, "n_sim_ge_0.8": 6, "acierto_1nn_en_train": 0.575, "alto_parecido_distinta_etiqueta": 0}, "test": {"sim_max_media": 0.525,  |
| review_status | warning | {"conteo": {"pendiente_revision_humana": 658}, "filas_no_revisadas": 658, "nota": "Resultados deben marcarse como sintéticos pendientes de validación."} |
| wayuu_status | ok | {"conteo": {"no_incluida_pendiente_validacion": 658}} |
| idioma | ok | {"conteo": {"es": 658}} |
| safety_flags_vs_intencion | warning | {"desviaciones_esperadas": {}, "otras_intenciones_con_bandera": {}, "mapeo": {"autorizacion": ["ninguna"], "cancelar_cita": ["ninguna"], "consultar_cita": ["ninguna"], "despedida": ["ninguna"], "estad |
| conflictos_lexicos_heuristicos | warning | {"urgencia_lexico_en_otra_intencion": [], "urgencia_sin_lexico": [{"id": "PQRS-0333", "text": "Requiero atención inmediata, presento dolor intenso en el pecho.", "intent": "urgencia_medica", "split":  |
| artefactos_generador | ok | {"cierre_por_intencion_max_exclamacion": 0.33, "palabras_por_estilo": {"coloquial": 7.0, "formal": 11.5, "simplificado": 4.4}, "palabras_por_intencion_min_max": [2.1, 9.7]} |
