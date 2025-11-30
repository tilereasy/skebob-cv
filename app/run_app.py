# app.py
import argparse

from main import main as run_pipeline
from dashboard import demo


def run_all():
    """
    1) Запускаем пайплайн (обработка видео, запись в БД)
    2) Поднимаем дашборд
    """
    print("[APP] Старт пайплайна...")
    run_pipeline()
    print("[APP] Пайплайн закончил работу, запускаем дашборд...")
    demo.launch(server_name="0.0.0.0", server_port=7860)


def run_only_pipeline():
    print("[APP] Запускаем только пайплайн (без дашборда)...")
    run_pipeline()


def run_only_dashboard():
    print("[APP] Запускаем только дашборд (без перерасчёта видео)...")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "pipeline", "dashboard"],
        default="all",
        help="all = сначала пайплайн, потом дашборд; "
             "pipeline = только обработка; "
             "dashboard = только дашборд",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all()
    elif args.mode == "pipeline":
        run_only_pipeline()
    elif args.mode == "dashboard":
        run_only_dashboard()
