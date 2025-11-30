# dashboard.py
"""
Интерактивный дашборд для визуализации активности людей на видео поезда.

ЛЕВАЯ ЧАСТЬ:
- Плеер (видео / размеченное видео / тепловая карта)
- Переключатель режима
- Слайдер текущей секунды
- Таблица "кто сейчас в кадре"

ПРАВАЯ ЧАСТЬ:
- Плашка с информацией о поезде
- KPI по видео в целом
- Галерея "опасных" моментов с таймкодами
- График: людей в кадре, работающих людей, индекс активности
  + вертикальная пунктирная линия по текущей секунде

db_app.py МЕНЯТЬ НЕЛЬЗЯ, только использовать.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from db_app import AsyncSessionLocal, Train, Second, SecondsPeople, People
from sqlalchemy import select


# -------------------------------------------------------------------
# ПУТИ И ФАЙЛЫ (ИСПОЛЬЗУЕМ РОВНО КАК ЗАДАНО)
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

# Кадры "опасных" моментов — по желанию можно сохранять сюда
DANGER_FRAMES_DIR = OUTPUT_DIR / "danger_frames"

# Режимы отображения медиа
VIDEO_MODES = ["Видео", "Размеченное видео", "Тепловая карта"]

# Кэш секунд по поездам, чтобы не бомбить БД
SECONDS_CACHE: Dict[str, pd.DataFrame] = {}


# -------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С БД
# -------------------------------------------------------------------

async def load_all_trains() -> List[Train]:
    """Загрузить все поезда."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Train).order_by(Train.id))
        trains = res.scalars().all()
    return list(trains)


async def get_train_by_number(train_number: str) -> Optional[Train]:
    """Получить поезд по номеру."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Train).where(Train.number == train_number)
        )
        train = res.scalar_one_or_none()
    return train


async def load_seconds_df(train_number: str) -> pd.DataFrame:
    """
    Загрузить все секунды для поезда в pandas.DataFrame.
    Результат кешируется.
    """
    if train_number in SECONDS_CACHE:
        return SECONDS_CACHE[train_number]

    async with AsyncSessionLocal() as db:
        res_train = await db.execute(
            select(Train).where(Train.number == train_number)
        )
        train = res_train.scalar_one_or_none()

        if not train:
            df_empty = pd.DataFrame(
                columns=[
                    "id",
                    "sequence_number",
                    "timestamp",
                    "people_count",
                    "active_people_count",
                    "activity_index",
                ]
            )
            SECONDS_CACHE[train_number] = df_empty
            return df_empty

        res_sec = await db.execute(
            select(Second)
            .where(Second.train_id == train.id)
            .order_by(Second.sequence_number)
        )
        seconds = res_sec.scalars().all()

    if not seconds:
        df_empty = pd.DataFrame(
            columns=[
                "id",
                "sequence_number",
                "timestamp",
                "people_count",
                "active_people_count",
                "activity_index",
            ]
        )
        SECONDS_CACHE[train_number] = df_empty
        return df_empty

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
    Вернуть DataFrame с людьми, которые есть в кадре на указанной секунде:
    - worker_type  (специальность из People.worker_type)
    - status       (статус активности из SecondsPeople.status)
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

    records = [
        {
            "person_id": person.id,
            "worker_type": person.worker_type,
            "status": sp.status or "",
        }
        for sp, person in rows
    ]

    if not records:
        return pd.DataFrame(columns=["person_id", "worker_type", "status"])

    return pd.DataFrame.from_records(records)


# -------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ФОРМАТИРОВАНИЯ / ВИЗУАЛИЗАЦИИ
# -------------------------------------------------------------------

def format_dt(dt) -> str:
    if dt is None:
        return "—"
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt)


def format_train_info(train: Optional[Train]) -> str:
    """Плашка с информацией о поезде."""
    if not train:
        return "### Информация о поезде\n\nНет данных по поезду."
    return (
        "### Информация о поезде\n\n"
        f"- Номер: **{train.number}**  \n"
        f"- Прибытие: **{format_dt(train.arrival_time)}**  \n"
        f"- Отбытие: **{format_dt(train.departure_time)}**"
    )


def build_kpi_markdown(train: Optional[Train], seconds_df: pd.DataFrame) -> str:
    """Собрать KPI по всему видео."""
    header = (
        f"### KPI по видео поезда {train.number}\n\n"
        if train
        else "### KPI по видео\n\n"
    )

    if seconds_df.empty:
        return header + "Нет данных по секундам для выбранного поезда."

    duration = int(seconds_df["sequence_number"].max())
    avg_people = float(seconds_df["people_count"].mean())
    avg_active = float(seconds_df["active_people_count"].mean())
    avg_activity = float(seconds_df["activity_index"].mean())
    max_activity = float(seconds_df["activity_index"].max())
    max_row = seconds_df.loc[seconds_df["activity_index"].idxmax()]
    max_activity_t = int(max_row["sequence_number"])

    text = (
        header
        + f"- Длительность видео: **{duration} с**  \n"
        + f"- Среднее число людей в кадре: **{avg_people:.1f}**  \n"
        + f"- Среднее число работающих людей: **{avg_active:.1f}**  \n"
        + f"- Средний индекс активности: **{avg_activity:.2f}**  \n"
        + f"- Пиковый индекс активности: **{max_activity:.2f}** на секунде **{max_activity_t}**"
    )
    return text


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
            yaxis_title="Значение",
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


def make_slider_update(seconds_df: pd.DataFrame) -> gr.Slider:
    """Вернуть gr.update(...) для слайдера по секундам."""
    if seconds_df.empty:
        return gr.update(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            interactive=False,
        )
    min_seq = int(seconds_df["sequence_number"].min())
    max_seq = int(seconds_df["sequence_number"].max())
    return gr.update(
        minimum=min_seq,
        maximum=max_seq,
        step=1,
        value=min_seq,
        interactive=True,
    )


def build_danger_highlights(
    train_number: str,
    seconds_df: pd.DataFrame,
    top_n: int = 6,
) -> List[Tuple[str, str]]:
    """
    Подборка "опасных" моментов.
    Сейчас критерий простой: топ-N секунд по activity_index.

    Ожидается наличие файлов:
      OUTPUT_DIR / "danger_frames" / {train_number} / sec_{sequence_number:06d}.jpg

    Возвращает: [(path, caption), ...]
    """
    if seconds_df.empty:
        return []

    df_sorted = seconds_df.sort_values(
        "activity_index", ascending=False
    ).head(top_n)

    items: List[Tuple[str, str]] = []

    for _, row in df_sorted.iterrows():
        seq = int(row["sequence_number"])
        tc = f"{seq // 60:02d}:{seq % 60:02d}"
        img_path = DANGER_FRAMES_DIR / train_number / f"sec_{seq:06d}.jpg"
        if not img_path.exists():
            # Если скрина нет — пропускаем
            continue
        caption = f"{tc} (t={seq} c, idx={row['activity_index']:.2f})"
        items.append((str(img_path), caption))

    return items


# -------------------------------------------------------------------
# ОБРАБОТЧИКИ GRADIO
# -------------------------------------------------------------------

async def on_app_load():
    """
    Инициализация интерфейса при запуске:
    - список поездов
    - выбор дефолтного поезда
    - слайдер секунд
    - KPI, опасные моменты, график, таблица людей на первой секунде
    """
    trains = await load_all_trains()

    # Заготовки на случай пустой БД
    if not trains:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Нет данных", template="plotly_white")

        empty_people_df = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )

        return (
            gr.update(choices=[], value=None, interactive=False),  # train_selector
            "### Информация о поезде\n\nВ базе нет поездов.",      # train_info
            gr.update(minimum=0, maximum=0, value=0, interactive=False),  # second_slider
            "### KPI\n\nНет данных по видео.",                     # kpi_markdown
            [],                                                    # danger_gallery
            empty_fig,                                             # activity_plot
            empty_people_df,                                       # people_table
        )

    # Есть поезда
    train_numbers = [t.number for t in trains]
    default_train = trains[0]
    default_number = default_train.number

    seconds_df = await load_seconds_df(default_number)
    slider_update = make_slider_update(seconds_df)

    # Текущая секунда (берём минимум, если есть данные)
    if seconds_df.empty:
        current_second = None
    else:
        current_second = int(seconds_df["sequence_number"].min())

    train_info_md = format_train_info(default_train)
    kpi_md = build_kpi_markdown(default_train, seconds_df)

    danger_items = build_danger_highlights(default_number, seconds_df)
    danger_value = [[path, caption] for path, caption in danger_items]

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
        ),                           # train_selector
        train_info_md,               # train_info
        slider_update,               # second_slider
        kpi_md,                      # kpi_markdown
        danger_value,                # danger_gallery
        activity_fig,                # activity_plot
        people_df,                   # people_table
    )


async def on_train_change(train_number: str):
    """
    Смена поезда:
    - обновить плашку поезда
    - пересчитать слайдер секунд
    - KPI, опасные моменты, график
    - таблицу людей для первой доступной секунды
    """
    if not train_number:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Нет данных", template="plotly_white")
        empty_people_df = pd.DataFrame(
            columns=["person_id", "worker_type", "status"]
        )
        return (
            "### Информация о поезде\n\nПоезд не выбран.",
            gr.update(minimum=0, maximum=0, value=0, interactive=False),
            "### KPI\n\nНет данных по видео.",
            [],
            empty_fig,
            empty_people_df,
        )

    train = await get_train_by_number(train_number)
    seconds_df = await load_seconds_df(train_number)

    train_info_md = format_train_info(train)
    slider_update = make_slider_update(seconds_df)
    kpi_md = build_kpi_markdown(train, seconds_df)

    if seconds_df.empty:
        current_second = None
    else:
        current_second = int(seconds_df["sequence_number"].min())

    danger_items = build_danger_highlights(train_number, seconds_df)
    danger_value = [[path, caption] for path, caption in danger_items]

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
        danger_value,    # danger_gallery
        activity_fig,    # activity_plot
        people_df,       # people_table
    )


async def on_second_change(
    train_number: str,
    second_value: float,
):
    """
    Перемотка видео (слайдер секунды):
    - обновить вертикальную линию на графике
    - обновить таблицу людей в кадре
    """
    second = int(second_value)
    seconds_df = await load_seconds_df(train_number)

    activity_fig = build_activity_figure(seconds_df, current_second=second)
    people_df = await get_people_for_second(
        train_number, second, seconds_df
    )

    return activity_fig, people_df


def on_mode_change(mode: str):
    """
    Переключение режима отображения:
    - Видео        -> INPUT_DIR/video.mov
    - Размеченное -> OUTPUT_DIR/result.mov
    - Тепловая    -> OUTPUT_DIR/heatmap.png (показываем картинкой)
    """
    if mode == "Видео":
        video_path = str(VIDEO_PATH) if VIDEO_PATH.exists() else None
        return (
            gr.update(value=video_path, visible=True),       # video_player
            gr.update(visible=False),                        # heatmap_image
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
        gr.update(visible=False),
        gr.update(value=heatmap_path, visible=True),
    )


# -------------------------------------------------------------------
# СБОРКА ИНТЕРФЕЙСА GRADIO
# -------------------------------------------------------------------

with gr.Blocks(title="Train Activity Dashboard") as demo:
    gr.Markdown("## Дашборд анализа активности на видео поезда")

    with gr.Row():
        # Выбор поезда
        train_selector = gr.Dropdown(
            label="Поезд",
            choices=[],
            value=None,
            interactive=True,
        )

        # Плашка с информацией о поезде
        train_info = gr.Markdown(
            value="### Информация о поезде\n\nЗагрузка...",
        )

    with gr.Row():
        # ЛЕВАЯ КОЛОНКА: видео / тепловая карта + управление + таблица людей
        with gr.Column(scale=3):
            # Плеер видео
            video_player = gr.Video(
                label="Видео",
                value=str(VIDEO_PATH) if VIDEO_PATH.exists() else None,
                interactive=False,
            )
            # Изображение тепловой карты (по умолчанию скрыто)
            heatmap_image = gr.Image(
                label="Тепловая карта",
                value=str(HEATMAP_PATH) if HEATMAP_PATH.exists() else None,
                visible=False,
            )

            # Переключатель режима воспроизведения
            mode_radio = gr.Radio(
                VIDEO_MODES,
                value="Видео",
                label="Режим отображения",
                interactive=True,
            )

            # Слайдер текущей секунды
            second_slider = gr.Slider(
                label="Текущая секунда видео",
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                interactive=False,
            )

            # Таблица с тем, что сейчас на экране
            people_table = gr.Dataframe(
                headers=["person_id", "worker_type", "status"],
                datatype=["number", "str", "str"],
                row_count=(0, "dynamic"),
                col_count=(3, "fixed"),
                label="Таблица с тем, что сейчас на экране",
            )

        # ПРАВАЯ КОЛОНКА: KPI, опасные моменты, график
        with gr.Column(scale=2):
            kpi_markdown = gr.Markdown(
                value="### KPI по видео\n\nЗагрузка...",
                label="KPI",
            )

            danger_gallery = gr.Gallery(
                label="Опасные моменты (скриншоты)",
                columns=3,
                rows=2,
                preview=True,
                height=220,
            )

            activity_plot = gr.Plot(
                label="График: количество людей, работающих людей и индекс активности",
            )

    # ----------------- СВЯЗЫВАНИЕ ОБРАБОТЧИКОВ -----------------

    # Инициализация при загрузке
    demo.load(
        fn=on_app_load,
        inputs=None,
        outputs=[
            train_selector,
            train_info,
            second_slider,
            kpi_markdown,
            danger_gallery,
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
            danger_gallery,
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

    # Переключение режима (видео / размеченное / тепловая карта)
    mode_radio.change(
        fn=on_mode_change,
        inputs=[mode_radio],
        outputs=[video_player, heatmap_image],
    )


if __name__ == "__main__":
    # queue=True позволяет нормально работать с async-обработчиками
    demo.queue().launch()
