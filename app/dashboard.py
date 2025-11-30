import os
from pathlib import Path
from typing import Tuple, Optional

import asyncio
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

# Импорт моделей и сессии из вашего db_app.py (не трогаем)
from db_app import AsyncSessionLocal, Train, Second, SecondsPeople, People
from sqlalchemy import select, func


# --------------------
# Пути и окружение
# --------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Имя файла видео
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

# Формат видео определяем по расширению файла.
if VIDEO_PATH.exists():
    VIDEO_FORMAT = VIDEO_PATH.suffix[1:].lower() or "mov"
else:
    VIDEO_FORMAT = "mov"

# Путь к медиа
VIDEO_PATHS = {
    "Видео": VIDEO_PATH,
    "Размеченное видео": LABELED_VIDEO_PATH,
    # для теплокарты будем использовать HEATMAP_PATH, но уже в gr.Image
}

# Поезд, для которого строим дашборд.
TRAIN_NUMBER = os.getenv("TRAIN_NUMBER", "Train-001")


# --------------------
# Загрузка данных из БД (агрегаты для графика/KPI)
# --------------------

async def _load_data_from_db(train_number: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные для дашборда из PostgreSQL по номеру поезда.
    """
    async with AsyncSessionLocal() as db:
        # 1) seconds
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
        sec_rows = res_sec.all()
        seconds_df = pd.DataFrame(
            sec_rows,
            columns=[
                "id",
                "sequence_number",
                "people_count",
                "active_people_count",
                "activity_index",
            ],
        )

        # 2) people
        q_people = (
            select(People.id, People.worker_type)
            .join(SecondsPeople, SecondsPeople.person_id == People.id)
            .join(Second, Second.id == SecondsPeople.second_id)
            .join(Train, Train.id == Second.train_id)
            .where(Train.number == train_number)
            .distinct()
        )
        res_people = await db.execute(q_people)
        people_rows = res_people.all()
        people_df = pd.DataFrame(people_rows, columns=["id", "worker_type"])

        # 3) seconds_people
        q_sp = (
            select(
                SecondsPeople.id,
                SecondsPeople.person_id,
                SecondsPeople.second_id,
                SecondsPeople.status,
            )
            .join(Second, Second.id == SecondsPeople.second_id)
            .join(Train, Train.id == Second.train_id)
            .where(Train.number == train_number)
        )
        res_sp = await db.execute(q_sp)
        sp_rows = res_sp.all()
        seconds_people_df = pd.DataFrame(
            sp_rows,
            columns=["id", "person_id", "second_id", "status"],
        )

    return seconds_df, people_df, seconds_people_df


async def _load_train_info(train_number: str) -> Optional[Train]:
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Train).where(Train.number == train_number))
        return res.scalar_one_or_none()


async def _init_all(train_number: str):
    seconds_df, people_df, seconds_people_df = await _load_data_from_db(train_number)
    train_obj = await _load_train_info(train_number)
    return seconds_df, people_df, seconds_people_df, train_obj


# Глобальные DataFrame'ы для графика/KPI + объект поезда.
try:
    SECONDS_DF, PEOPLE_DF, SECONDS_PEOPLE_DF, TRAIN_OBJ = asyncio.run(_init_all(TRAIN_NUMBER))
    print(
        f"Loaded data for train '{TRAIN_NUMBER}': "
        f"{len(SECONDS_DF)} seconds, "
        f"{len(PEOPLE_DF)} people, "
        f"{len(SECONDS_PEOPLE_DF)} seconds_people."
    )
except Exception as e:
    print(f"Failed to load data from DB for train '{TRAIN_NUMBER}': {e}")
    SECONDS_DF = pd.DataFrame(
        columns=["id", "sequence_number", "people_count", "active_people_count", "activity_index"]
    )
    PEOPLE_DF = pd.DataFrame(columns=["id", "worker_type"])
    SECONDS_PEOPLE_DF = pd.DataFrame(columns=["id", "person_id", "second_id", "status"])
    TRAIN_OBJ = None


# --------------------
# Функции для отображения
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


async def _table_at_time_async(train_number: str, t_sec: float) -> pd.DataFrame:
    """
    1. sequence_number ~= t_sec
    2. находим Second для этого поезда
    3. по second_id берём SecondsPeople + People
    """
    if SECONDS_DF.empty:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    seq_min = int(SECONDS_DF["sequence_number"].min())
    seq_max = int(SECONDS_DF["sequence_number"].max())

    seq = int(round(float(t_sec or 0.0)))
    seq = max(seq_min, min(seq, seq_max))

    async with AsyncSessionLocal() as db:
        q_second = (
            select(Second.id)
            .join(Train, Train.id == Second.train_id)
            .where(Train.number == train_number, Second.sequence_number == seq)
        )
        res_second = await db.execute(q_second)
        second_id = res_second.scalar_one_or_none()

        if second_id is None:
            return pd.DataFrame(columns=["id", "worker_type", "status"])

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


def table_at_time(t_sec: float) -> pd.DataFrame:
    return asyncio.run(_table_at_time_async(TRAIN_NUMBER, t_sec))


def update_by_time(current_t: float):
    t = float(current_t or 0.0)

    fig = make_main_figure(SECONDS_DF)

    try:
        tbl = table_at_time(t)
    except Exception as e:
        print(f"Failed to build table_at_time({t}): {e}")
        tbl = pd.DataFrame(columns=["id", "worker_type", "status"])

    return fig, tbl, t


# --- НОВАЯ функция переключения между видео и картинкой ---  # NEW
def switch_media(mode: str):
    """
    Переключает:
      - 'Видео' / 'Размеченное видео' -> gr.Video
      - 'Тепловая карта'             -> gr.Image
    """
    # дефолт: всё скрыто
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
# ----------------------------------------------------------


# --------------------
# Предварительно построенные объекты
# --------------------

INIT_FIG = make_main_figure(SECONDS_DF)
INIT_TABLE = table_at_time(0.0)
INIT_KPI = kpi_markdown(SECONDS_DF)
TRAIN_INFO_MD = train_info_markdown(TRAIN_OBJ)


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
            # Видео (по умолчанию видно)                         # CHANGED: добавили visible
            video = gr.Video(
                value=str(VIDEO_PATH) if VIDEO_PATH.exists() else None,
                format=VIDEO_FORMAT,
                label="Видео",
                elem_id="video_player",
                visible=True,
            )

            # Картинка теплокарты (по умолчанию скрыта)          # NEW
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
                value=INIT_TABLE,
                label="Таблица с тем, что сейчас на экране",
                interactive=False,
                wrap=True,
            )

        with gr.Column(scale=2):
            train_box = gr.Markdown(value=TRAIN_INFO_MD)
            kpi_box = gr.Markdown(value=INIT_KPI)

            with gr.Accordion("Danger highlights (картинки)", open=True):
                danger_gallery = gr.Gallery(
                    label="",
                    columns=3,
                    height="auto",
                )

            with gr.Accordion("График людей, работающих людей и индекса активности", open=True):
                main_plot = gr.Plot(
                    value=INIT_FIG,
                )

    timer = gr.Timer(0.5)

    timer.tick(
        js=READ_VIDEO_TIME_JS,
        outputs=current_time,
    )

    current_time.change(
        fn=update_by_time,
        inputs=current_time,
        outputs=[main_plot, table_now, time_state],
    )

    # Переключение режима: видео <-> теплокарта              # CHANGED
    mode_radio.change(
        fn=switch_media,
        inputs=mode_radio,
        outputs=[video, heatmap_image],
    )


if __name__ == "__main__":
    # VIDEO_DIR у тебя уже нет – используем INPUT/OUTPUT      # CHANGED
    demo.launch(
        debug=True,
        allowed_paths=[str(INPUT_DIR), str(OUTPUT_DIR)],
    )
