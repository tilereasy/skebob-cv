# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ARG APP_ROOT=/workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ROOT=${APP_ROOT}

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

WORKDIR ${APP_ROOT}

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY infra/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN mkdir -p data/input data/output datasets models && \
    chmod +x /usr/local/bin/entrypoint.sh

ENV APP_MODE=all \
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/postgres \
    SKIP_DB_BOOTSTRAP=0 \
    PYTHONPATH=${APP_ROOT}:${PYTHONPATH}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
