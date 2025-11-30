import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from db_app import AsyncSessionLocal, Train, Second, SecondsPeople, People
from sqlalchemy import select

# --------------------
# Пути и окружение
# --------------------

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

print("APP_DIR      =", APP_DIR)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("INPUT_DIR    =", INPUT_DIR)
print("OUTPUT_DIR   =", OUTPUT_DIR)
print("VIDEO_PATH   =", VIDEO_PATH, "| exists:", VIDEO_PATH.exists())

if VIDEO_PATH.exists():
    VIDEO_FORMAT = VIDEO_PATH.suffix[1:].lower() or "mov"
else:
    VIDEO_FORMAT = "mov"

VIDEO_PATHS = {
    "Видео": VIDEO_PATH,
    "Размеченное видео": LABELED_VIDEO_PATH,
    # "Тепловая карта" — отдельный gr.Image, см. ниже
}

TRAIN_NUMBER = os.getenv("TRAIN_NUMBER", "Train-001")


# --------------------
# Вспомогательные функции рендера
# --------------------

def make_main_figure(seconds_df: pd.DataFrame) -> go.Figure:
    if seconds_df.empty:
        return go.Figure()

    x = seconds_df["sequence_number"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=seconds_df["people_count"],
            mode="lines",
            name="Людей в кадре",
            line=dict(color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=seconds_df["active_people_count"],
            mode="lines",
            name="Работающих людей",
            line=dict(color="#2ca02c"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=seconds_df["activity_index"],
            mode="lines",
            name="Индекс активности",
            yaxis="y2",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )

    fig.update_layout(
        title=f"Поезд {TRAIN_NUMBER}: люди, работающие люди и индекс активности",
        xaxis=dict(title="Время, сек (sequence_number)"),
        yaxis=dict(title="Количество людей"),
        yaxis2=dict(
            title="Индекс активности",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def kpi_markdown(seconds_df: pd.DataFrame) -> str:
    if seconds_df.empty:
        return f"Нет данных по поезду **{TRAIN_NUMBER}**"

    total_seconds = seconds_df["sequence_number"].nunique()
    duration_min = total_seconds // 60
    duration_sec = total_seconds % 60

    max_active = seconds_df["active_people_count"].max()
    min_active = seconds_df["active_people_count"].min()
    avg_activity = seconds_df["activity_index"].mean()

    md = f"""
**KPI по видео (поезд {TRAIN_NUMBER})**

- Длительность видео: {total_seconds} сек (~{duration_min} мин {duration_sec} сек)
- Минимум работающих людей в кадре за всё видео: {min_active}
- Максимум работающих людей в кадре за всё видео: {max_active}
- Средний индекс активности: {avg_activity:.2f}
"""
    return md


def train_info_markdown(train: Optional[Train]) -> str:
    if train is None:
        return f"Поезд с номером **{TRAIN_NUMBER}** не найден в базе данных."

    def fmt_dt(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "не задано"

    arrival = fmt_dt(train.arrival_time)
    departure = fmt_dt(train.departure_time)

    md = f"""
### Информация о поезде

- Номер поезда: **{train.number}**
- Время прибытия: {arrival}
- Время отправления: {departure}
"""
    return md


# --------------------
# Работа с БД (всё async, без asyncio.run)
# --------------------

async def load_seconds_df(db, train_number: str) -> pd.DataFrame:
    """
    seconds_df для поезда:
    - id
    - sequence_number
    - people_count
    - active_people_count
    - activity_index
    """
    q_sec = (
        select(
            Second.id,
            Second.sequence_number,
            Second.people_count,
            Second.active_people_count,
            Second.activity_index,
        )
        .join(Train, Train.id == Second.train_id)
        .where(Train.number == train_number)
        .order_by(Second.sequence_number)
    )
    res_sec = await db.execute(q_sec)
    rows = res_sec.all()
    return pd.DataFrame(
        rows,
        columns=["id", "sequence_number", "people_count", "active_people_count", "activity_index"],
    )


async def load_train(db, train_number: str) -> Optional[Train]:
    res = await db.execute(select(Train).where(Train.number == train_number))
    return res.scalar_one_or_none()


async def table_at_time_async(db, train_number: str, t_sec: float) -> pd.DataFrame:
    """
    Алгоритм:
      1. Определяем整数 sequence_number ~ t_sec (>=1).
      2. Находим Second для этого поезда и sequence_number.
      3. По second_id берём SecondsPeople + People.
      4. Возвращаем DataFrame[id, worker_type, status].
    """
    # sequence_number: просто int(round(t_sec)), минимально 1
    seq = int(round(float(t_sec or 0.0)))
    if seq < 1:
        seq = 1

    # 1) Находим second_id для этого поезда и sequence_number
    q_second = (
        select(Second.id)
        .join(Train, Train.id == Second.train_id)
        .where(Train.number == train_number, Second.sequence_number == seq)
    )
    res_second = await db.execute(q_second)
    second_id = res_second.scalar_one_or_none()

    if second_id is None:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    # 2) Люди в эту секунду
    q_people_at_time = (
        select(People.id, People.worker_type, SecondsPeople.status)
        .join(SecondsPeople, SecondsPeople.person_id == People.id)
        .where(SecondsPeople.second_id == second_id)
        .order_by(People.id)
    )
    res_people = await db.execute(q_people_at_time)
    rows = res_people.all()

    if not rows:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    df = pd.DataFrame(rows, columns=["id", "worker_type", "status"])
    df = df.sort_values("id").reset_index(drop=True)
    return df


# --------------------
# Переключение видео / теплокарты
# --------------------

def switch_media(mode: str):
    """
    Переключает:
      - 'Видео' / 'Размеченное видео' -> gr.Video
      - 'Тепловая карта'             -> gr.Image
    """
    video_update = gr.update(visible=False)
    heatmap_update = gr.update(visible=False)

    if mode in ["Видео", "Размеченное видео"]:
        path = VIDEO_PATHS.get(mode, VIDEO_PATH)
        if path is not None and Path(path).exists():
            video_update = gr.update(
                value=str(path),
                visible=True,
            )
        else:
            video_update = gr.update(value=None, visible=True)
        heatmap_update = gr.update(visible=False)

    elif mode == "Тепловая карта":
        video_update = gr.update(visible=False)
        if HEATMAP_PATH.exists():
            heatmap_update = gr.update(
                value=str(HEATMAP_PATH),
                visible=True,
            )
        else:
            heatmap_update = gr.update(value=None, visible=True)

    return video_update, heatmap_update


# --------------------
# Основной callback: ВСЁ из БД по текущему времени
# --------------------

async def update_by_time(current_t: float):
    """
    Вызывается на каждый тик таймера.
    Делает:
      - seconds_df для поезда -> график и KPI;
      - second по sequence_number ~ current_t -> таблица людей;
      - Train -> плашка поезда.
    """
    t = float(current_t or 0.0)

    async with AsyncSessionLocal() as db:
        # seconds (для графика и KPI)
        seconds_df = await load_seconds_df(db, TRAIN_NUMBER)

        # таблица людей на текущую секунду
        table_df = await table_at_time_async(db, TRAIN_NUMBER, t)

        # информация о поезде
        train_obj = await load_train(db, TRAIN_NUMBER)

    fig = make_main_figure(seconds_df)
    kpi_md = kpi_markdown(seconds_df)
    train_md = train_info_markdown(train_obj)

    # возвращаем:
    # - график
    # - таблицу
    # - текущее время (state)
    # - KPI
    # - плашку поезда
    return fig, table_df, t, kpi_md, train_md


# --------------------
# Предварительные значения (пока без БД)
# --------------------

EMPTY_TABLE = pd.DataFrame(columns=["id", "worker_type", "status"])
EMPTY_FIG = go.Figure()
INIT_KPI = f"Загрузка данных по поезду **{TRAIN_NUMBER}**..."
INIT_TRAIN_MD = f"Загрузка информации о поезде **{TRAIN_NUMBER}**..."


# --------------------
# JS: синхронизация с видео
# --------------------

READ_VIDEO_TIME_JS = """
() => {
  let video = null;

  const root = document.querySelector('#video_player');
  if (root) {
    const tag = (root.tagName || '').toLowerCase();
    if (tag === 'video') {
      video = root;
    } else {
      video = root.querySelector('video');
    }
  }

  if (!video) {
    video = document.querySelector('video');
  }

  if (!video) {
    return 0;
  }

  return video.currentTime || 0;
}
"""


# --------------------
# UI
# --------------------

with gr.Blocks() as demo:
    time_state = gr.State(0.0)
    current_time = gr.Number(value=0.0, visible=False, label="current_time_sync")

    with gr.Row():
        with gr.Column(scale=3):
            # Видео по умолчанию
            video = gr.Video(
                value=str(VIDEO_PATH) if VIDEO_PATH.exists() else None,
                format=VIDEO_FORMAT,
                label="Видео",
                elem_id="video_player",
                visible=True,
            )

            # Картинка теплокарты (скрыта по умолчанию)
            heatmap_image = gr.Image(
                value=str(HEATMAP_PATH) if HEATMAP_PATH.exists() else None,
                label="Тепловая карта",
                visible=False,
            )

            mode_radio = gr.Radio(
                choices=["Видео", "Размеченное видео", "Тепловая карта"],
                value="Видео",
                label="Режим воспроизведения",
            )

            table_now = gr.Dataframe(
                headers=["id", "worker_type", "status"],
                value=EMPTY_TABLE,
                label="Таблица с тем, что сейчас на экране",
                interactive=False,
                wrap=True,
            )

        with gr.Column(scale=2):
            train_box = gr.Markdown(value=INIT_TRAIN_MD)
            kpi_box = gr.Markdown(value=INIT_KPI)

            with gr.Accordion("Danger highlights (картинки)", open=True):
                danger_gallery = gr.Gallery(
                    label="",
                    columns=3,
                    height="auto",
                )

            with gr.Accordion("График людей, работающих людей и индекса активности", open=True):
                main_plot = gr.Plot(
                    value=EMPTY_FIG,
                )

    # Таймер
    timer = gr.Timer(0.5)

    timer.tick(
        js=READ_VIDEO_TIME_JS,
        outputs=current_time,
    )

    # async callback: Gradio сам await-ит update_by_time
    current_time.change(
        fn=update_by_time,
        inputs=current_time,
        outputs=[main_plot, table_now, time_state, kpi_box, train_box],
    )

    # Переключение режимов: видео / теплокарта
    mode_radio.change(
        fn=switch_media,
        inputs=mode_radio,
        outputs=[video, heatmap_image],
    )


if __name__ == "__main__":
    demo.launch(
        debug=True,
        allowed_paths=[str(INPUT_DIR), str(OUTPUT_DIR)],
    )
