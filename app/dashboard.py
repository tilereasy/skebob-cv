# dashboard.py
"""
Дашборд визуализации активности людей на видео поезда.

Схема компоновки:

СЛЕВА:
- Проигрыватель видео (оригинал / размеченное / тепловая карта)
- Переключатель режима
- Слайдер текущей секунды
- Таблица "кто сейчас в кадре"

СПРАВА:
- Плашка о поезде (номер, прибытие, отбытие)
- KPI по видео
- Таблица "опасных" моментов (таймкоды самых высоких activity_index)
- График: всего людей, работающих людей, индекс активности
  + вертикальная пунктирная линия на текущей секунде

db_app.py менять НЕЛЬЗЯ — только используем его модели и AsyncSessionLocal.
"""

import os
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from db_app import AsyncSessionLocal, Train, Second, SecondsPeople, People
from sqlalchemy import select


# -------------------------------------------------------------------
# ПУТИ К ДАННЫМ (ИСПОЛЬЗУЕМ РОВНО КАК ЗАДАНО)
# -------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Файлы медиа
VIDEO_FILENAME = "video.mov"
VIDEO_PATH = INPUT_DIR / VIDEO_FILENAME

LABELED_VIDEO_FILENAME = "result.mov"
LABELED_VIDEO_PATH = OUTPUT_DIR / LABELED_VIDEO_FILENAME

HEATMAP_FILENAME = "heatmap.png"
HEATMAP_PATH = OUTPUT_DIR / HEATMAP_FILENAME

# Режимы воспроизведения
VIDEO_MODES = ["Видео", "Размеченное видео", "Тепловая карта"]

# Кэш секунд по поездам (ключ — номер поезда)
SECONDS_CACHE: Dict[str, pd.DataFrame] = {}


# -------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ БД
# -------------------------------------------------------------------

async def load_all_trains():
    """Получить список всех поездов из БД."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Train).order_by(Train.id))
        return list(res.scalars().all())


async def get_train_by_number(train_number: str) -> Optional[Train]:
    """Получить объект Train по его номеру."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Train).where(Train.number == train_number))
        return res.scalar_one_or_none()


async def load_seconds_df(train_number: str) -> pd.DataFrame:
    """
    Загрузить все секунды для поезда с указанным номером в DataFrame.
    Результат кешируется в SECONDS_CACHE.
    """
    if train_number in SECONDS_CACHE:
        return SECONDS_CACHE[train_number]

    async with AsyncSessionLocal() as db:
        # Берём все секунды по join-у с поездом по номеру
        res = await db.execute(
            select(Second)
            .join(Train, Second.train_id == Train.id)
            .where(Train.number == train_number)
            .order_by(Second.sequence_number)
        )
        seconds = list(res.scalars().all())

    if not seconds:
        df = pd.DataFrame(
            columns=[
                "id",
                "sequence_number",
                "timestamp",
                "people_count",
                "active_people_count",
                "activity_index",
            ]
        )
        SECONDS_CACHE[train_number] = df
        return df

    df = pd.DataFrame(
        [
            {
                "id": s.id,
                "sequence_number": s.sequence_number,
                "timestamp": s.timestamp,
                "people_count": s.people_count,
                "active_people_count": s.active_people_count,
                "activity_index": s.activity_index,
            }
            for s in seconds
        ]
    )
    SECONDS_CACHE[train_number] = df
    return df


async def get_people_for_second(
    train_number: str,
    sequence_number: int,
    seconds_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Получить людей, которые находятся в кадре на указанной секунде видео
    данного поезда.

    Возвращает DataFrame с колонками:
    - person_id
    - worker_type (специальность)
    - status      (активность)
    """
    if seconds_df is None:
        seconds_df = await load_seconds_df(train_number)

    if seconds_df.empty:
        return pd.DataFrame(columns=["person_id", "worker_type", "status"])

    row = seconds_df[seconds_df["sequence_number"] == sequence_number]
    if row.empty:
        return pd.DataFrame(columns=["person_id", "worker_type", "status"])

    second_id = int(row["id"].iloc[0])

    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(SecondsPeople, People)
            .join(People, SecondsPeople.person_id == People.id)
            .where(SecondsPeople.second_id == second_id)
        )
        rows = res.all()

    people_records = [
        {
            "person_id": person.id,
            "worker_type": person.worker_type,
            "status": sp.status or "",
        }
        for sp, person in rows
    ]

    if not people_records:
        return pd.DataFrame(columns=["person_id", "worker_type", "status"])

    return pd.DataFrame.from_records(people_records)


# -------------------------------------------------------------------
# ФОРМАТИРОВАНИЕ ТЕКСТА / ГРАФИКОВ
# -------------------------------------------------------------------

def format_dt(dt) -> str:
    if dt is None:
        return "—"
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt)


def format_train_info_md(train: Optional[Train]) -> str:
    """Markdown-плашка с информацией о поезде."""
    if not train:
        return "### Информация о поезде\n\nПоезд не выбран или отсутствует в БД."

    return (
        "### Информация о поезде\n\n"
        f"- Номер: **{train.number}**  \n"
        f"- Время прибытия: **{format_dt(train.arrival_time)}**  \n"
        f"- Время отбытия: **{format_dt(train.departure_time)}**"
    )


def build_kpi_md(train: Optional[Train], seconds_df: pd.DataFrame) -> str:
    """Собрать KPI по видео в виде Markdown."""
    header = (
        f"### KPI по видео поезда {train.number}\n\n"
        if train
        else "### KPI по видео\n\n"
    )

    if seconds_df.empty:
        return header + "Нет данных по секундам для выбранного поезда."

    duration_sec = int(seconds_df["sequence_number"].max())
    avg_people = float(seconds_df["people_count"].mean())
    avg_active = float(seconds_df["active_people_count"].mean())
    avg_activity = float(seconds_df["activity_index"].mean())
    max_activity = float(seconds_df["activity_index"].max())
    max_row = seconds_df.loc[seconds_df["activity_index"].idxmax()]
    max_t = int(max_row["sequence_number"])

    return (
        header
        + f"- Длительность видео: **{duration_sec} с**  \n"
        + f"- Среднее количество людей в кадре: **{avg_people:.1f}**  \n"
        + f"- Среднее количество работающих людей: **{avg_active:.1f}**  \n"
        + f"- Средний индекс активности: **{avg_activity:.2f}**  \n"
        + f"- Пиковый индекс активности: **{max_activity:.2f}** на секунде **{max_t}**"
    )


def build_danger_table(seconds_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Таблица "опасных" моментов — топ-N секунд с максимальным activity_index.

    Колонки:
    - second          (секунда видео)
    - timecode        (MM:SS)
    - activity_index
    - people_count
    - active_people_count
    """
    if seconds_df.empty:
        return pd.DataFrame(
            columns=[
                "second",
                "timecode",
                "activity_index",
                "people_count",
                "active_people_count",
            ]
        )

    df = seconds_df.copy()
    df = df.sort_values("activity_index", ascending=False).head(top_n)

    def to_tc(s: int) -> str:
        return f"{s // 60:02d}:{s % 60:02d}"

    out = pd.DataFrame(
        {
            "second": df["sequence_number"].astype(int),
            "timecode": df["sequence_number"].astype(int).map(to_tc),
            "activity_index": df["activity_index"],
            "people_count": df["people_count"],
            "active_people_count": df["active_people_count"],
        }
    )

    return out.reset_index(drop=True)


def build_activity_figure(
    seconds_df: pd.DataFrame,
    current_second: Optional[int] = None,
) -> go.Figure:
    """График: всего людей, работающих людей и индекс активности."""
    fig = go.Figure()

    if seconds_df.empty:
        fig.update_layout(
            title="Нет данных по секундам для выбранного поезда",
            xaxis_title="Секунда видео",
            yaxis_title="Значения",
            template="plotly_white",
        )
        return fig

    x = seconds_df["sequence_number"]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=seconds_df["people_count"],
            name="Всего людей",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=seconds_df["active_people_count"],
            name="Работающие люди",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=seconds_df["activity_index"],
            name="Индекс активности",
            mode="lines",
            line=dict(dash="dot"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis_title="Секунда видео",
        yaxis=dict(
            title="Количество людей",
            side="left",
        ),
        yaxis2=dict(
            title="Индекс активности",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        legend=dict(orientation="h", x=0, y=1.15),
        margin=dict(l=50, r=60, t=40, b=40),
        height=320,
    )

    if current_second is not None:
        fig.add_vline(
            x=current_second,
            line_dash="dash",
            line_color="black",
            annotation_text=f"t={current_second} c",
            annotation_position="top right",
        )

    return fig


def slider_update_for_seconds(seconds_df: pd.DataFrame):
    """Сконфигурировать слайдер по данным секунд."""
    if seconds_df.empty:
        return gr.update(
            minimum=0,
            maximum=0,
            value=0,
            step=1,
            interactive=False,
        )

    min_seq = int(seconds_df["sequence_number"].min())
    max_seq = int(seconds_df["sequence_number"].max())
    return gr.update(
        minimum=min_seq,
        maximum=max_seq,
        value=min_seq,
        step=1,
        interactive=True,
    )


# -------------------------------------------------------------------
# ОБРАБОТЧИКИ GRADIO (ASYNC)
# -------------------------------------------------------------------

async def on_app_load():
    """
    Инициализация интерфейса при загрузке:
    - заполняем список поездов
    - выбираем первый поезд
    - подгружаем по нему секунды, KPI, график, "опасные" моменты
    - таблицу людей для первой секунды
    """
    trains = await load_all_trains()

    # Если в БД нет поездов
    if not trains:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Нет данных", template="plotly_white")

        empty_danger = pd.DataFrame(
            columns=[
                "second",
                "timecode",
                "activity_index",
                "people_count",
                "active_people_count",
            ]
        )
        empty_people = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )

        return (
            gr.update(choices=[], value=None, interactive=False),  # train_selector
            "### Информация о поезде\n\nВ базе нет поездов.",      # train_info
            gr.update(                                            # second_slider
                minimum=0, maximum=0, value=0, step=1, interactive=False
            ),
            "### KPI по видео\n\nНет данных.",                    # kpi_markdown
            empty_danger,                                         # danger_table
            empty_fig,                                            # activity_plot
            empty_people,                                         # people_table
        )

    # Поезда есть
    train_numbers = [t.number for t in trains]
    default_train = trains[0]
    default_number = default_train.number

    seconds_df = await load_seconds_df(default_number)
    slider_update = slider_update_for_seconds(seconds_df)

    if seconds_df.empty:
        current_second = None
    else:
        current_second = int(seconds_df["sequence_number"].min())

    train_info_md = format_train_info_md(default_train)
    kpi_md = build_kpi_md(default_train, seconds_df)
    danger_df = build_danger_table(seconds_df)
    activity_fig = build_activity_figure(seconds_df, current_second)

    if current_second is not None:
        people_df = await get_people_for_second(
            default_number, current_second, seconds_df
        )
    else:
        people_df = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )

    return (
        gr.update(
            choices=train_numbers,
            value=default_number,
            interactive=True,
        ),                       # train_selector
        train_info_md,           # train_info
        slider_update,           # second_slider
        kpi_md,                  # kpi_markdown
        danger_df,               # danger_table
        activity_fig,            # activity_plot
        people_df,               # people_table
    )


async def on_train_change(train_number: str):
    """
    Смена поезда:
    - обновить информацию о поезде
    - пересчитать диапазон слайдера
    - KPI, опасные моменты, график
    - таблицу людей для первой секунды
    """
    if not train_number:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Нет данных", template="plotly_white")

        empty_danger = pd.DataFrame(
            columns=[
                "second",
                "timecode",
                "activity_index",
                "people_count",
                "active_people_count",
            ]
        )
        empty_people = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )

        return (
            "### Информация о поезде\n\nПоезд не выбран.",
            gr.update(
                minimum=0, maximum=0, value=0, step=1, interactive=False
            ),
            "### KPI по видео\n\nНет данных.",
            empty_danger,
            empty_fig,
            empty_people,
        )

    train = await get_train_by_number(train_number)
    seconds_df = await load_seconds_df(train_number)

    train_info_md = format_train_info_md(train)
    slider_update = slider_update_for_seconds(seconds_df)
    kpi_md = build_kpi_md(train, seconds_df)
    danger_df = build_danger_table(seconds_df)

    if seconds_df.empty:
        current_second = None
    else:
        current_second = int(seconds_df["sequence_number"].min())

    activity_fig = build_activity_figure(seconds_df, current_second)

    if current_second is not None:
        people_df = await get_people_for_second(
            train_number, current_second, seconds_df
        )
    else:
        people_df = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )

    return (
        train_info_md,   # train_info
        slider_update,   # second_slider
        kpi_md,          # kpi_markdown
        danger_df,       # danger_table
        activity_fig,    # activity_plot
        people_df,       # people_table
    )


async def on_second_change(train_number: str, second_value: float):
    """
    Перемотка по слайдеру текущей секунды:
    - двигаем вертикальную линию на графике
    - обновляем таблицу людей в кадре
    """
    if not train_number:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Нет данных", template="plotly_white")
        empty_people = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )
        return empty_fig, empty_people

    seq = int(second_value)
    seconds_df = await load_seconds_df(train_number)

    activity_fig = build_activity_figure(seconds_df, current_second=seq)
    people_df = await get_people_for_second(train_number, seq, seconds_df)

    return activity_fig, people_df


def on_mode_change(mode: str):
    """
    Переключение режима воспроизведения:

    - "Видео"           -> INPUT_DIR / "video.mov"
    - "Размеченное..."  -> OUTPUT_DIR / "result.mov"
    - "Тепловая карта"  -> OUTPUT_DIR / "heatmap.png" (отображаем как картинку)
    """
    if mode == "Видео":
        video_path = str(VIDEO_PATH) if VIDEO_PATH.exists() else None
        return (
            gr.update(value=video_path, visible=True),  # video_player
            gr.update(visible=False),                   # heatmap_image
        )

    if mode == "Размеченное видео":
        labeled_path = (
            str(LABELED_VIDEO_PATH) if LABELED_VIDEO_PATH.exists() else None
        )
        return (
            gr.update(value=labeled_path, visible=True),
            gr.update(visible=False),
        )

    # Тепловая карта
    heatmap_path = str(HEATMAP_PATH) if HEATMAP_PATH.exists() else None
    return (
        gr.update(visible=False),                      # video_player
        gr.update(value=heatmap_path, visible=True),   # heatmap_image
    )


# -------------------------------------------------------------------
# СБОРКА ИНТЕРФЕЙСА GRADIO
# -------------------------------------------------------------------

with gr.Blocks(title="Train Activity Dashboard") as demo:
    gr.Markdown("## Дашборд анализа активности на видео поезда")

    # Верхний ряд: выбор поезда + плашка поезда
    with gr.Row():
        train_selector = gr.Dropdown(
            label="Поезд",
            choices=[],
            value=None,
            interactive=True,
        )

        train_info = gr.Markdown(
            value="### Информация о поезде\n\nЗагрузка данных...",
        )

    # Основной блок: слева видео и таблица, справа KPI и графики
    with gr.Row():
        # Левая колонка: медиа + слайдер + таблица людей
        with gr.Column(scale=3):
            video_player = gr.Video(
                label="Видео",
                value=str(VIDEO_PATH) if VIDEO_PATH.exists() else None,
                interactive=False,
            )

            heatmap_image = gr.Image(
                label="Тепловая карта",
                value=str(HEATMAP_PATH) if HEATMAP_PATH.exists() else None,
                visible=False,
            )

            mode_radio = gr.Radio(
                VIDEO_MODES,
                value="Видео",
                label="Режим отображения",
                interactive=True,
            )

            second_slider = gr.Slider(
                label="Текущая секунда видео",
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                interactive=False,
            )

            people_table = gr.Dataframe(
                headers=["person_id", "worker_type", "status"],
                datatype=["number", "str", "str"],
                row_count=(0, "dynamic"),
                col_count=(3, "fixed"),
                label="Таблица с тем, что сейчас на экране",
            )

        # Правая колонка: KPI, опасные моменты, график
        with gr.Column(scale=2):
            kpi_markdown = gr.Markdown(
                value="### KPI по видео\n\nЗагрузка...",
                label="KPI",
            )

            danger_table = gr.Dataframe(
                headers=[
                    "second",
                    "timecode",
                    "activity_index",
                    "people_count",
                    "active_people_count",
                ],
                datatype=["number", "str", "number", "number", "number"],
                row_count=(0, "dynamic"),
                col_count=(5, "fixed"),
                label="Опасные моменты (по activity_index)",
            )

            activity_plot = gr.Plot(
                label="График: количество людей, работающих людей и индекс активности",
            )

    # ----------------- ПРИВЯЗКА ОБРАБОТЧИКОВ -----------------

    # Инициализация при загрузке страницы
    demo.load(
        fn=on_app_load,
        inputs=None,
        outputs=[
            train_selector,
            train_info,
            second_slider,
            kpi_markdown,
            danger_table,
            activity_plot,
            people_table,
        ],
    )

    # Смена поезда
    train_selector.change(
        fn=on_train_change,
        inputs=[train_selector],
        outputs=[
            train_info,
            second_slider,
            kpi_markdown,
            danger_table,
            activity_plot,
            people_table,
        ],
    )

    # Перемотка по секундам
    second_slider.change(
        fn=on_second_change,
        inputs=[train_selector, second_slider],
        outputs=[activity_plot, people_table],
    )

    # Переключение режима воспроизведения (видео / размеченное / тепловая карта)
    mode_radio.change(
        fn=on_mode_change,
        inputs=[mode_radio],
        outputs=[video_player, heatmap_image],
    )


# -------------------------------------------------------------------
# ТОЧКА ВХОДА
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Без queue() — максимально простой запуск.
    # Запуск: python dashboard.py
    demo.launch()
