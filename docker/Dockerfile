# syntax=docker/dockerfile:1
# GreenForge historical replay only: no ETL, notebooks, or training jobs in Docker.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src
WORKDIR /app

# A small one-shot image: copy + rebase Windows-origin MLflow metadata.
FROM base AS mlflow-init
COPY docker/prepare_mlflow.py /opt/greenforge/prepare_mlflow.py
CMD ["python", "/opt/greenforge/prepare_mlflow.py"]

# API image: only the source files needed by /predict, report reading, and MLflow.
FROM base AS api
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements-api.txt \
    && rm /tmp/requirements-api.txt

# Do NOT copy setup.py or the big notebook/training requirements.txt.
COPY app/main.py /app/app/main.py
COPY src/greenforge_ai/training/train_forecasting.py /app/src/greenforge_ai/training/train_forecasting.py
COPY src/greenforge_ai/training/validation.py /app/src/greenforge_ai/training/validation.py
COPY src/greenforge_ai/training/evaluation.py /app/src/greenforge_ai/training/evaluation.py
COPY src/greenforge_ai/features/interaction_features.py /app/src/greenforge_ai/features/interaction_features.py
COPY src/greenforge_ai/ingestion/dwd_client.py /app/src/greenforge_ai/ingestion/dwd_client.py

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin greenforge \
    && mkdir -p /app/logs/application \
    && chown -R greenforge:greenforge /app/logs
USER greenforge
ENV HOME=/home/greenforge
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Dashboard image: no sklearn, LightGBM, MLflow, optimization or ETL packages.
FROM base AS dashboard
COPY docker/requirements-dashboard.txt /tmp/requirements-dashboard.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements-dashboard.txt \
    && rm /tmp/requirements-dashboard.txt
COPY app/streamlit_app.py /app/app/streamlit_app.py
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin greenforge \
    && mkdir -p /home/greenforge/.streamlit \
    && chown -R greenforge:greenforge /home/greenforge/.streamlit
USER greenforge
ENV HOME=/home/greenforge \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
