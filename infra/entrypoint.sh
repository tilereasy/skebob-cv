#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/workspace}"
MODE="${APP_MODE:-all}"

export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"
cd "${APP_ROOT}"

if [[ "${SKIP_DB_BOOTSTRAP:-0}" != "1" ]]; then
  python - <<'PY'
import asyncio
import os

import asyncpg
from app.db_app import create_tables

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is not set")

ASYNC_PG = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)


async def wait_for_db():
    last_error = None
    for attempt in range(30):
        try:
            conn = await asyncpg.connect(ASYNC_PG)
        except Exception as exc:  # pragma: no cover - bootstrap utility
            last_error = exc
            print(f"[entrypoint] DB not ready (attempt {attempt + 1}/30): {exc}")
            await asyncio.sleep(2)
        else:
            await conn.close()
            return
    raise RuntimeError(f"Database never became ready: {last_error}")


async def bootstrap():
    await wait_for_db()
    await create_tables()
    print("[entrypoint] Database is ready")


asyncio.run(bootstrap())
PY
fi

exec python "${APP_ROOT}/app/run_app.py" --mode "${MODE}"
