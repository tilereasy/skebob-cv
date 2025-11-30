# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY mosh.py ./mosh.py
COPY infra/entrypoint.sh ./infra/entrypoint.sh

RUN mkdir -p data/input data/output datasets models && \
    chmod +x infra/entrypoint.sh

ENV APP_MODE=all \
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/postgres \
    SKIP_DB_BOOTSTRAP=0

ENTRYPOINT ["bash", "infra/entrypoint.sh"]
