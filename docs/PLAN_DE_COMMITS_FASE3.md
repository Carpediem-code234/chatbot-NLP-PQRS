# Plan de commits — rama `FASE-3-Integracion-y-Despliegue`

Igual que en la Fase 2: un paso por sesión de trabajo, en días distintos, probando antes de cada commit. No cambies las fechas.

## 0. Crear la rama desde la Fase 2

```bash
cd ~/Escritorio/chatbot-NLP-PQRS
git checkout FASE-2-Desarrollo-y-Modelado && git pull
git checkout -b FASE-3-Integracion-y-Despliegue
# Descomprime el .zip encima (agrega app/, docs/, Dockerfile… y actualiza README, requirements y .gitignore)
source .venv/bin/activate
pip install -r requirements.txt
python -m src.entrenar --rapido      # el bot necesita models/modelo_pqrs_m3.joblib
python -m pytest -q                  # 79 pruebas
git status                           # NO debe aparecer .env ni data/pqrs.db
```

## 1. Dependencias, configuración y almacenamiento

```bash
git add requirements.txt .gitignore .env.example app/__init__.py app/config.py app/almacen.py
git commit -m "feat(app): configuración por variables de entorno y almacenamiento SQLite de PQRS"
```

## 2. Reglas de seguridad, mensajes y diálogo

```bash
git add app/seguridad.py app/mensajes.py app/dialogo.py app/simular.py
git commit -m "feat(dialogo): flujo de conversación PQRS con confirmación, menú, radicado y reglas de urgencia"
```

## 3. API FastAPI y webhook de WhatsApp

```bash
git add app/whatsapp.py app/main.py
git commit -m "feat(whatsapp): webhook con verificación de firma, envío por Cloud API y endpoints de administración"
```

## 4. Pruebas

```bash
python -m pytest -q
git add tests/test_bot_whatsapp.py
git commit -m "test(app): reglas de seguridad, diálogo, webhook y API"
```

## 5. Despliegue y documentación

```bash
git add Dockerfile .dockerignore docs/DESPLIEGUE_WHATSAPP.md README.md
git commit -m "docs(despliegue): guía de WhatsApp Business, Dockerfile y README de la Fase 3"
```

## 6. Después de la prueba real con WhatsApp

Cuando conectes el número de prueba y verifiques la conversación, agrega evidencia (capturas en `docs/evidencias/`, sin números de teléfono visibles):

```bash
git add docs/evidencias/
git commit -m "docs: evidencias de la prueba con WhatsApp real"
```

Publicar cuando quieras: `git push -u origin FASE-3-Integracion-y-Despliegue`.
