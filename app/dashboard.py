import os
from pathlib import Path
from typing import Tuple

import asyncio
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

# Импорт моделей и сессии из вашего db_app.py
from db_app import AsyncSessionLocal, Train, Second, SecondsPeople, People
from sqlalchemy import select, func


# --------------------
# Пути и окружение
# --------------------

# Явно выключаем прокси, чтобы Gradio / браузер не пытались
# ходить наружу через системные прокси.
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(var, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "output"

# Имя файла видео; при необходимости замените на свой.
VIDEO_FILENAME = "result.mov"
VIDEO_PATH = VIDEO_DIR / VIDEO_FILENAME

print("APP_DIR      =", APP_DIR)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("VIDEO_DIR    =", VIDEO_DIR)
print("VIDEO_PATH   =", VIDEO_PATH, "| exists:", VIDEO_PATH.exists())

# Формат видео определяем по расширению файла.
if VIDEO_PATH.exists():
    VIDEO_FORMAT = VIDEO_PATH.suffix[1:].lower() or "mov"
else:
    VIDEO_FORMAT = "mov"

# Все режимы пока указывают на один и тот же файл;
# при необходимости подставьте свои пути.
VIDEO_PATHS = {
    "Видео": VIDEO_PATH,
    "Размеченное видео": VIDEO_PATH,
    "Тепловая карта": VIDEO_PATH,
}

# Поезд, для которого строим дашборд.
# Можно вынести в настройки / UI, пока фиксируем константой.
TRAIN_NUMBER = os.getenv("TRAIN_NUMBER", "Train-001")


# --------------------
# Загрузка данных из БД
# --------------------

async def _load_data_from_db(train_number: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные для дашборда из PostgreSQL (через db_app.py)
    по номеру поезда.

    Возвращает
    ----------
    seconds_df : pd.DataFrame
        - id                (PK seconds)
        - sequence_number   (порядковый номер секунды для поезда)
        - people_count
        - active_people_count
        - activity_index

    people_df : pd.DataFrame
        - id
        - worker_type

    seconds_people_df : pd.DataFrame
        - id
        - person_id
        - second_id
        - status
    """
    async with AsyncSessionLocal() as db:
        # 1) seconds для выбранного поезда
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
            columns=["id", "sequence_number", "people_count", "active_people_count", "activity_index"],
        )

        # 2) people, которые вообще когда‑либо появлялись у этого поезда
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

        # 3) seconds_people только для секунд этого поезда
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
            sp_rows, columns=["id", "person_id", "second_id", "status"]
        )

    return seconds_df, people_df, seconds_people_df


# Синхронная обёртка, чтобы загрузить данные один раз при старте модуля.
try:
    SECONDS_DF, PEOPLE_DF, SECONDS_PEOPLE_DF = asyncio.run(_load_data_from_db(TRAIN_NUMBER))
    print(f"Loaded data for train '{TRAIN_NUMBER}': "
          f"{len(SECONDS_DF)} seconds, {len(PEOPLE_DF)} people, {len(SECONDS_PEOPLE_DF)} seconds_people.")
except Exception as e:
    print(f"Failed to load data from DB for train '{TRAIN_NUMBER}': {e}")
    SECONDS_DF = pd.DataFrame(columns=["id", "sequence_number", "people_count", "active_people_count", "activity_index"])
    PEOPLE_DF = pd.DataFrame(columns=["id", "worker_type"])
    SECONDS_PEOPLE_DF = pd.DataFrame(columns=["id", "person_id", "second_id", "status"])


# --------------------
# Функции для отображения
# --------------------

def make_main_figure(seconds_df: pd.DataFrame) -> go.Figure:
    """
    Построение основного графика на основе таблицы seconds:

    - people_count               -> "Людей в кадре"
    - active_people_count        -> "Работающих людей"
    - activity_index (от 0 до 1) -> "Индекс активности" (вторая ось Y)
    """
    if seconds_df.empty:
        return go.Figure()

    # Используем sequence_number как ось времени (секунды от начала поезда/ролика).
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


def table_at_time(
    people_df: pd.DataFrame,
    seconds_df: pd.DataFrame,
    seconds_people_df: pd.DataFrame,
    t_sec: float,
) -> pd.DataFrame:
    """
    Таблица под проигрывателем на момент времени t_sec.

    Формат:
        - id человека
        - его специальность (worker_type)
        - его статус в данный момент (status)

    Маппинг времени:
        - предполагаем, что sequence_number соответствует секунде видео;
        - округляем t_sec до ближайшего целого sequence_number;
        - находим second_id с таким sequence_number;
        - по second_id вытаскиваем записи из seconds_people.
    """
    if people_df.empty or seconds_df.empty or seconds_people_df.empty:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    # Рабочий диапазон по sequence_number
    seq_min = int(seconds_df["sequence_number"].min())
    seq_max = int(seconds_df["sequence_number"].max())

    seq = int(round(t_sec))
    seq = max(seq_min, min(seq, seq_max))

    # Ищем second_id, соответствующий этому sequence_number
    row = seconds_df.loc[seconds_df["sequence_number"] == seq]
    if row.empty:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    second_id = int(row["id"].iloc[0])

    sub = seconds_people_df[seconds_people_df["second_id"] == second_id]
    if sub.empty:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    merged = sub.merge(people_df, left_on="person_id", right_on="id", how="left")

    result = (
        merged[["person_id", "worker_type", "status"]]
        .rename(columns={"person_id": "id"})
        .sort_values("id")
        .reset_index(drop=True)
    )

    return result


def kpi_markdown(seconds_df: pd.DataFrame) -> str:
    """
    KPI по видео:

    - длительность видео;
    - минимум / максимум active_people_count;
    - средний activity_index.
    """
    if seconds_df.empty:
        return "Нет данных по этому поезду"

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


def update_by_time(current_t: float):
    """
    Callback, который обновляет:
        - основную фигуру (агрегированный график по seconds);
        - таблицу под проигрывателем (статус людей в текущий момент).

    current_t — текущее время видео (секунды) с фронтенда.
    """
    t = float(current_t or 0.0)

    fig = make_main_figure(SECONDS_DF)
    tbl = table_at_time(PEOPLE_DF, SECONDS_DF, SECONDS_PEOPLE_DF, t)

    return fig, tbl, t


def mode_to_video_value(mode: str):
    """
    Маппинг выбранного режима воспроизведения (radio) на путь к видеофайлу.
    """
    path = VIDEO_PATHS.get(mode, VIDEO_PATH)
    if path is not None and Path(path).exists():
        return str(path)
    return None


# Предварительно построенный график, таблица и KPI.
INIT_FIG = make_main_figure(SECONDS_DF)
INIT_TABLE = table_at_time(PEOPLE_DF, SECONDS_DF, SECONDS_PEOPLE_DF, 0.0)
INIT_KPI = kpi_markdown(SECONDS_DF)


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
    # Состояние с текущим временем на стороне сервера.
    time_state = gr.State(0.0)

    # Текущее время видео, синхронизируемое с фронтендом (через JS + Timer).
    current_time = gr.Number(value=0.0, visible=False, label="current_time_sync")

    with gr.Row():
        with gr.Column(scale=3):
            # Видеоплеер. elem_id нужен для поиска в JS.
            video = gr.Video(
                value=str(VIDEO_PATH) if VIDEO_PATH.exists() else None,
                format=VIDEO_FORMAT,
                label="Видео",
                elem_id="video_player",
            )

            # Переключатель режима отображения видео (сырое / разметка / тепловая карта).
            mode_radio = gr.Radio(
                choices=["Видео", "Размеченное видео", "Тепловая карта"],
                value="Видео",
                label="Режим воспроизведения",
            )

            # Таблица с людьми, присутствующими в кадре в текущий момент времени:
            # id человека, его специальность (worker_type), его статус.
            table_now = gr.Dataframe(
                headers=["id", "worker_type", "status"],
                value=INIT_TABLE,
                label="Таблица с тем, что сейчас на экране",
                interactive=False,
                wrap=True,
            )

        with gr.Column(scale=2):
            # KPI по ролику (длительность, min/max активных людей, средний индекс активности).
            kpi_box = gr.Markdown(value=INIT_KPI)

            # Галерея потенциально "опасных" кадров (placeholder).
            with gr.Accordion("Danger highlights (картинки)", open=True):
                danger_gallery = gr.Gallery(
                    label="",
                    columns=3,
                    height="auto",
                )

            # Основной график: люди / работающие / индекс активности.
            with gr.Accordion("График людей, работающих людей и индекса активности", open=True):
                main_plot = gr.Plot(
                    value=INIT_FIG,
                )

    # Таймер, который каждые 0.5 секунды опрашивает текущее время видео на фронтенде.
    timer = gr.Timer(0.5)

    # 1) Таймер вызывает JS, который читает video.currentTime и пишет его в current_time.
    timer.tick(
        js=READ_VIDEO_TIME_JS,
        outputs=current_time,  # результат JS -> value скрытого компонента current_time
    )

    # 2) При каждом изменении current_time на Python-стороне
    #    пересчитываем график и таблицу (update_by_time).
    current_time.change(
        fn=update_by_time,
        inputs=current_time,                 # текущее время видео (секунды)
        outputs=[main_plot, table_now, time_state],
    )

    # Переключение режима воспроизведения: подменяем источник видео.
    mode_radio.change(
        fn=mode_to_video_value,
        inputs=mode_radio,
        outputs=video,
    )


if __name__ == "__main__":
    demo.launch(debug=True, allowed_paths=[str(VIDEO_DIR)])
