FROM python:3.13-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY app/ app/
COPY data/raw/ data/raw/
# El modelo se entrena al construir la imagen (≈15 s): no hace falta versionar el .joblib
RUN python -m src.entrenar --rapido
ENV DB_PATH=/srv/datos/pqrs.db
VOLUME ["/srv/datos"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
