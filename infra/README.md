# Инфраструктура

Все docker-артефакты лежат в этой директории. Основной сценарий — поднять PostgreSQL и CV-приложение (пайплайн + Gradio-дашборд) одной командой.

## Файлы

- `Dockerfile` — образ с Python 3.11, системными зависимостями (ffmpeg, opencv и т.д.) и установленными python-зависимостями из `requirements.txt`.
- `entrypoint.sh` — точка входа контейнера. Ждет готовности базы, создаёт таблицы через `app.db_app.create_tables()` и запускает `python app/run_app.py --mode <APP_MODE>`.
- `docker-compose.yml` — описывает два сервиса: `postgres` и `cv-app`. Внутрь CV-контейнера примаплены `../data`, `../datasets`, `../models`.
- `.env.example` — шаблон переменных окружения. При необходимости скопируйте в `.env` и измените значения.

## Быстрый старт

```bash
cp infra/.env.example infra/.env  # опционально, чтобы переопределить настройки
docker compose -f infra/docker-compose.yml up --build
```

После первого запуска пайплайн обработает входное видео, а дашборд будет доступен на `http://localhost:7860`.

## Полезные переменные

Все переменные берутся из `infra/.env` (если файл существует) или окружения:

- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — настройки БД.
- `DATABASE_URL` — строка подключения для SQLAlchemy/asyncpg внутри приложения.
- `APP_MODE` — режим `run_app.py` (`all`, `pipeline` или `dashboard`).
- `APP_HTTP_PORT` — порт Gradio (пробрасывается наружу).
- `SKIP_DB_BOOTSTRAP` — установите `1`, если не нужно создавать таблицы при старте контейнера (например, когда БД управляется другими средствами).

## Использование GPU

По умолчанию контейнер работает на CPU. Если требуется GPU, стартуйте стек с параметром Docker `--gpus=all`, например:

```bash
docker compose -f infra/docker-compose.yml run --gpus all cv-app
```

Либо добавьте `"device_requests"` в `docker-compose.yml` под свои нужды.

## Обновление зависимостей

При изменении `requirements.txt` выполните пересборку:

```bash
docker compose -f infra/docker-compose.yml build cv-app
```

## Пример полезных команд

- `docker compose -f infra/docker-compose.yml up --build` — запуск стека с пересборкой образа.
- `docker compose -f infra/docker-compose.yml up cv-app` — только приложение (при условии, что PostgreSQL уже запущен отдельно, либо переменная `SKIP_DB_BOOTSTRAP=1`).
- `docker compose -f infra/docker-compose.yml down -v` — остановка сервисов и удаление volume с данными PostgreSQL.
