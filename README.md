# Skebob CV Pipeline

Комплекс из CV-пайплайна и дашборда для анализа активности на железнодорожной площадке. Основной сценарий: обработка входного видео (`app/main.py`) → запись агрегатов в PostgreSQL → визуализация метрик и событий в Gradio-дешборде (`app/dashboard.py`). В репозитории есть весь код и инфраструктура (Docker, docker-compose) для локального развёртывания.

## Разработка в рамках хакатона «Сибинтек‑Софт»
Этот репозиторий создан в рамках трека
«Хакатон по искусственному интеллекту “СИБИНТЕК‑СОФТ”» компании «Сибинтек‑Софт».

Команда: t34M4 n3 R4k0V

Участники:
[Listy-V](https://github.com/Listy-V),

[Kwerdu](https://github.com/Kwerd-u),

[Zama47](https://github.com/Zama47),

[tilereasy](https://github.com/tilereasy).


Проект разрабатывался в ограниченные сроки как proof‑of‑concept:

фокус на функциональности и исследовании идей, а не на полном промышленном продакшене;
код и архитектура ориентированы на быструю итерацию.

При внедрении в боевую среду рекомендуется:

* провести аудит качества и безопасности кода;
* настроить CI/CD и мониторинг;
* адаптировать конфигурацию под инфраструктуру компании
(логирование, трассировка, секреты, политика доступа к данным и т.д.).

## Возможности

- YOLO-детекция поездов и людей, классификация ролей и отслеживание треков (см. `app/main.py`).
- Подсчёт людей, активности и событий по секундам с сохранением в PostgreSQL (async SQLAlchemy).
- OCR поезда через EasyOCR и сохранение логов.
- Gradio-дешборд для просмотра KPI, графиков активности, «опасных» кадров и текущих состояний.
- Папки с моделями, датасетами и результатами (`data/input`, `data/output`, `models`, `datasets`).
- Dockerfile + docker-compose для быстрого старта без ручного управления зависимостями.

## Структура проекта

```
app/
  main.py           # основной пайплайн (обработка видео, запись в БД, OCR и т.д.)
  run_app.py        # CLI для запуска пайплайна, дашборда или обоих
  dashboard.py      # Gradio-дешборд
  db_app.py         # async SQLAlchemy, модели, CRUD-операции
data/
  input/            # входное видео (по умолчанию data/input/video.mp4)
  output/           # результат обработки, csv-логи, тепловая карта, алерты
datasets/           # тренировочные/вспомогательные датасеты
infra/
  Dockerfile        # образ CV-приложения
  docker-compose.yml# стек postgres + cv-app
  entrypoint.sh     # ожидание БД, создание таблиц и запуск приложения
models/             # веса YOLO и конфиг трекера
requirements.txt    # Python-зависимости
```

## Требования

- Python 3.11+ (для выполнения без Docker).
- PostgreSQL 13+ (используется asyncpg, строка подключения задаётся переменной `DATABASE_URL`).
- Системные библиотеки для OpenCV, ffmpeg и Tesseract (если работаете вне Docker, установите вручную).
- GPU не обязателен, но ускоряет инференс YOLO. В Docker его можно пробросить через `--gpus all`.

## Установка и запуск (локально без Docker)

1. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Убедитесь, что PostgreSQL запущен и доступен. Пример строки подключения (значение по умолчанию):
   ```bash
   export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
   ```
3. Один раз создайте таблицы:
   ```bash
   python - <<'PY'
   import asyncio
   from app.db_app import create_tables
   asyncio.run(create_tables())
   PY
   ```
4. Запуск скриптов:
   - Только пайплайн: `python app/run_app.py --mode pipeline`
   - Только дашборд: `python app/run_app.py --mode dashboard`
   - Пайплайн -> дашборд (последовательно): `python app/run_app.py --mode all`

По умолчанию дашборд слушает `http://0.0.0.0:7860`. Настройте `HOST`/`PORT`, если нужно, через Gradio (`demo.launch`).

## Запуск через Docker

В директории `infra/` лежит полный стек.

1. (Опционально) скопируйте и измените переменные окружения:
   ```bash
   cp infra/.env.example infra/.env
   # отредактируйте DATABASE_URL, APP_MODE, APP_HTTP_PORT и т.д.
   ```
2. Соберите и запустите:
   ```bash
   docker compose -f infra/docker-compose.yml up --build
   ```
   Это поднимет `postgres` и `cv-app`. `entrypoint.sh` дождётся готовности БД, создаст таблицы и запустит `python app/run_app.py --mode ${APP_MODE}` (по умолчанию `all`).
3. Весь проект монтируется в `/workspace`, поэтому пайплайн использует локальные данные (`data/`), модели и любые правки кода. Для GPU добавьте `--gpus all`:
   ```bash
   docker compose -f infra/docker-compose.yml run --gpus all cv-app
   ```
4. Завершение работы:
   ```bash
   docker compose -f infra/docker-compose.yml down    # остановить
   docker compose -f infra/docker-compose.yml down -v # остановить и удалить volume с данными PostgreSQL
   ```

Больше деталей — в `infra/README.md`.

## Переменные окружения

- `DATABASE_URL` — строка подключения для SQLAlchemy/asyncpg (формат `postgresql+asyncpg://user:pass@host:port/db`).
- `APP_MODE` (`all`, `pipeline`, `dashboard`) — управляет поведением `app/run_app.py`.
- `TRAIN_NUMBER`, `VIDEO_PATH` и др. — смотрите в `app/main.py`/`app/dashboard.py`, можно переопределять через окружение или аргументы.
- `SKIP_DB_BOOTSTRAP` (Docker) — если `1`, контейнер не создаёт таблицы на старте.

## Типичный сценарий работы

1. Поместите входное видео в `data/input/video.mp4` и веса моделей в `models/`.
2. Запустите пайплайн — он обработает видео, создаст `data/output/result.mp4`, CSV-логи, тепловую карту и запишет показатели в PostgreSQL.
3. Поднимите дашборд (`--mode dashboard`), выберите поезд в списке — увидите KPI, графики, «опасные» моменты и таблицу актуальных людей.
4. При необходимости добавляйте собственные метрики / визуализации в `app/dashboard.py` или дорабатывайте пайплайн в `app/main.py`.


## Тестирование и отладка

- Для проверки отдельных компонентов (например, БД) есть утилиты в `app/db_app.py`.
- В дашборде предусмотрена обработка отсутствующих данных (нет CSV, пустые таблицы и т.д.), логируйте ошибки через `print` или `logging`.
- Если изменяете зависимости в `requirements.txt`, пересоберите Docker-образ (`docker compose -f infra/docker-compose.yml build cv-app`).

## Полезные команды

- `python -m app.main` — прямой запуск пайплайна (если нужно обойти `run_app.py`).
- `python -m app.dashboard` — запуск только Gradio-приложения.
- `docker compose -f infra/docker-compose.yml logs -f` — просмотр логов контейнеров.
- `docker compose -f infra/docker-compose.yml up cv-app` — только приложение (если PostgreSQL уже запущен отдельно, а `DATABASE_URL` указывает на него).

## Обратная связь

Если нужно расширить функциональность (например, добавить новые модели, метрики или интеграции), смело редактируйте соответствующие модули. Главное — не ломайте основной пайплайн в `app/main.py`, как и просил автор проекта :)
