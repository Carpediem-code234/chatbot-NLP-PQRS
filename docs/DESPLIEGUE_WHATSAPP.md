# Despliegue del chatbot en WhatsApp Business (Cloud API)

Guía paso a paso para conectar el bot a un número de WhatsApp real. Tiempo estimado: 30–45 minutos la primera vez.

```
WhatsApp del usuario ──► Meta (Cloud API) ──POST /webhook──► nuestro servidor (FastAPI)
                                                                │ reglas + modelo M3 + SQLite
WhatsApp del usuario ◄── Meta (Graph API) ◄──POST /messages─────┘
```

## 0. Requisitos

- Cuenta en **Meta for Developers** (<https://developers.facebook.com>) y un portafolio de **Meta Business**.
- Un celular con WhatsApp para recibir las pruebas.
- El proyecto instalado (`pip install -r requirements.txt`) y el modelo entrenado (`python -m src.entrenar`).
- Una **URL pública con HTTPS** que apunte a tu servidor (paso 4).

## 1. Crear la app de Meta y obtener el número de prueba

1. developers.facebook.com → **Mis apps → Crear app** → caso de uso **"Conectarte con los clientes a través de WhatsApp"**.
2. En **WhatsApp → Configuración de la API** (API Setup) verás:
   - un **número de prueba** de Meta (remitente),
   - el **Identificador del número de teléfono** → `PHONE_NUMBER_ID`,
   - el botón **Generar token de acceso** → token temporal (dura unas 24 h; sirve para la primera prueba).
3. En **"Para" (To)** agrega tu celular y confírmalo con el código. Con el número de prueba solo se puede escribir a **hasta 5 números** registrados así: agrega los de los evaluadores.

## 2. Token permanente (para que el bot no deje de funcionar al día siguiente)

1. business.facebook.com → **Configuración del negocio → Usuarios → Usuarios del sistema → Agregar** (rol Administrador).
2. **Asignar activos** → tu app → control total.
3. **Generar token** → selecciona la app y los permisos `whatsapp_business_messaging` y `whatsapp_business_management`.
4. Copia el token → `WA_TOKEN`.

## 3. Archivo `.env`

```bash
cp .env.example .env
```

Completa:

| Variable | De dónde sale |
|---|---|
| `WA_TOKEN` | paso 2 (o el temporal del paso 1 para probar) |
| `PHONE_NUMBER_ID` | paso 1 |
| `VERIFY_TOKEN` | **lo inventas tú** (p. ej. `pqrs-guajira-2026`); lo escribirás igual en Meta |
| `APP_SECRET` | Configuración de la app → **Básica → Clave secreta de la app** |
| `ADMIN_TOKEN` | lo inventas tú; protege `/api/pqrs` |

`.env` está en `.gitignore`: **nunca lo subas al repositorio**.

## 4. Levantar el servidor con una URL pública

### Opción A · En tu computador + túnel (para la demo)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

En otra terminal, expón el puerto con HTTPS. Cualquiera de estas dos:

```bash
ngrok http 8000                                    # requiere cuenta gratuita en ngrok.com
cloudflared tunnel --url http://localhost:8000     # sin cuenta; URL *.trycloudflare.com
```

Copia la URL `https://…` que aparece. **Cambia cada vez que reinicias el túnel**: si cambia, actualízala en Meta (paso 5). El computador debe quedar encendido durante la demo.

### Opción B · En un servidor (para que funcione sin tu computador)

Con Docker en cualquier servidor o plataforma que lo acepte (Render, Railway, Fly.io, una VM):

```bash
docker build -t pqrs-bot .
docker run -d -p 8000:8000 --env-file .env -v pqrs-datos:/srv/datos pqrs-bot
```

El volumen `pqrs-datos` conserva la base de datos de PQRS entre reinicios. En plataformas con disco efímero, sin volumen se pierden las PQRS al reiniciar.

Comprueba que responde: `https://TU-URL/health` → `{"ok": true}`.

## 5. Configurar el webhook en Meta

1. App → **WhatsApp → Configuración** → **Webhook → Editar**.
2. **URL de devolución de llamada**: `https://TU-URL/webhook`
3. **Token de verificación**: el mismo `VERIFY_TOKEN` del `.env`.
4. **Verificar y guardar**. Meta hace un `GET /webhook` y el servidor responde el reto (en la consola verás `GET /webhook … 200`).
5. En **Campos del webhook**, **suscríbete a `messages`**. Sin esto Meta no envía los mensajes.

## 6. Probar

Desde un celular registrado en el paso 1, escribe al número de prueba:

```
hola
no me han entregado la insulina de mi mamá
1
12345678
```

El bot debe pedir confirmación, luego el documento, y responder con un número de radicado `PQRS-AAAAMMDD-NNNNN`. Más casos para la demo:

| Escribe | Qué debe pasar |
|---|---|
| `mi bebe convulsiona` | aviso de emergencia (123), en cualquier momento de la conversación |
| `agente` | aviso de que un funcionario escribirá |
| `lo de la otra vez` | menú 1-4 (el modelo no está seguro) |
| `jamaya, mi hijo ayuulii y no hay tottoolu` | saludo en wayuunaiki + queja |
| `KIERO SITA PAL DOTOR` | petición de cita |
| `como va el PQRS-…` | estado del radicado |
| una nota de voz | explica que aún no escucha audios |
| `cancelar` | reinicia la conversación |

## 7. Gestionar las PQRS (funcionario)

Documentación interactiva en `https://TU-URL/docs`: cada endpoint de administración tiene un campo `x-admin-token` donde pegas tu `ADMIN_TOKEN`. Desde la terminal:

```bash
curl -H "X-Admin-Token: $ADMIN_TOKEN" https://TU-URL/api/pqrs
curl -H "X-Admin-Token: $ADMIN_TOKEN" https://TU-URL/api/pqrs/resumen
curl -H "X-Admin-Token: $ADMIN_TOKEN" https://TU-URL/api/eventos?tipo=urgencia
curl -X PATCH -H "X-Admin-Token: $ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d '{"estado": "respondida", "respuesta": "Ya puede reclamar su medicamento en la farmacia."}' \
     https://TU-URL/api/pqrs/PQRS-20261119-00001
```

El `PATCH` le avisa al usuario por WhatsApp.

## Problemas frecuentes

| Síntoma | Causa probable |
|---|---|
| Meta dice que no pudo validar la URL | `VERIFY_TOKEN` distinto, servidor apagado o URL sin `https`/sin `/webhook` |
| El bot recibe pero no responde; en consola `WhatsApp respondió 401` | token vencido (usa el permanente del paso 2) |
| `WhatsApp respondió 400 … recipient not in allowed list` | el celular no está registrado en "Para" (paso 1.3) |
| No llega nada al servidor | no te suscribiste al campo `messages` (paso 5.5) o cambió la URL del túnel |
| `firma inválida` en consola | `APP_SECRET` incorrecto |
| El aviso del `PATCH` no llega | WhatsApp solo permite mensajes libres **dentro de las 24 h** siguientes al último mensaje del usuario; fuera de esa ventana se necesita una plantilla aprobada (no implementado) |

## Limitaciones conocidas

- Número de prueba: máximo 5 destinatarios. Para atender usuarios reales se necesita un número propio verificado y la verificación del negocio en Meta.
- Las notas de voz no se transcriben todavía (se registran como evento `audio`).
- La regla de urgencias es una lista de frases: **no reemplaza el triage médico**.
- Las respuestas en wayuunaiki se limitan al saludo, y el vocabulario debe validarse con hablantes nativos.
- SQLite basta para el prototipo; con varios servidores a la vez se necesitaría PostgreSQL.
