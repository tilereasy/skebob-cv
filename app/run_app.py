"""
app.py — единая точка входа:

1) Запускает видеопайплайн из main.py (YOLO + запись в БД, CSV, heatmap и т.п.)
2) После завершения обработки поднимает Gradio-дэшборд из dashboard.py.

Этот файл удобно использовать как ENTRYPOINT / CMD в Dockerfile.
"""

import os

# импортируем main() из твоего пайплайна
from main import main as run_pipeline

# импортируем Gradio-приложение (объект demo) из dashboard.py
from dashboard import demo


def main():
    """
    Высокоуровневый сценарий:
    1. При необходимости прогоняем пайплайн.
    2. Стартуем дашборд.
    """

    # Можно управлять, запускать ли обработку при старте контейнера
    # через переменную окружения, чтобы не гонять пайплайн каждый раз.
    run_processing = os.getenv("RUN_PIPELINE_ON_START", "1") == "1"

    if run_processing:
        print("[APP] Запускаю видеопайплайн (main.main())...")
        # main() внутри сам использует asyncio.run(...) для БД — это окей.
        run_pipeline()
        print("[APP] Пайплайн завершён, данные записаны в БД и файлы.")

    else:
        print("[APP] Пропускаю запуск пайплайна (RUN_PIPELINE_ON_START != '1').")

    print("[APP] Стартую Gradio-дэшборд...")
    # Можно переопределять порт/адрес через переменные окружения, удобно в Docker.
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "7860"))

    # demo — это объект gr.Blocks из dashboard.py
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
