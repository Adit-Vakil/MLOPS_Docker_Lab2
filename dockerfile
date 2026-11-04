FROM python:3.10-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS model_training

COPY src/model_training.py .
RUN python model_training.py

FROM base AS serving

COPY --from=model_training /app/wine_model.joblib .
COPY --from=model_training /app/metrics.json .
COPY src/main.py .
COPY src/templates/ ./templates/

EXPOSE 4000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:4000/health', timeout=2).close()"]

CMD ["python", "main.py"]
