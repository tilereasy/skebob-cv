import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr


# --------------------
# Пути и окружение
# --------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Файлы медиа
VIDEO_FILENAME = "video.mp4"
VIDEO_PATH = INPUT_DIR / VIDEO_FILENAME

LABELED_VIDEO_FILENAME = "result.mp4"
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
}

# Один фиксированный поезд
TRAIN_NUMBER = "ЭП20 076"


# --------------------
# СИНТЕТИЧЕСКИЕ ДАННЫЕ (вместо БД)
# --------------------

def build_seconds_df() -> pd.DataFrame:
    """
    Генерируем правдоподобные данные по секундам:
      - длительность 12 минут = 720 секунд;
      - active_people_count: мин 0, макс 3;
      - activity_index: среднее ровно 0.55;
      - people_count >= active_people_count.
    """
    total_seconds = 12 * 60  # 12 минут
    seq = np.arange(1, total_seconds + 1)

    # Активные люди: повторяющийся паттерн 0-1-2-3-2-1
    pattern_active = np.array([0, 1, 2, 3, 2, 1])
    active_people_count = np.tile(
        pattern_active,
        int(np.ceil(total_seconds / len(pattern_active)))
    )[:total_seconds]

    # Всего людей в кадре: всегда >= active_people_count
    # Немного "шума" через детерминированную формулу
    extra_people = (np.arange(total_seconds) * 7) % 3  # 0,1,2,0,1,2,...
    people_count = active_people_count + extra_people

    # Индекс активности: "волнистый" вокруг 0.55, среднее ровно 0.55
    activity_index = np.empty(total_seconds, dtype=float)
    half = total_seconds // 2
    for k in range(half):
        delta = 0.10 * np.sin((k + 1) / 20.0)  # амплитуда 0.1
        i1 = 2 * k
        i2 = 2 * k + 1
        activity_index[i1] = 0.55 + delta
        activity_index[i2] = 0.55 - delta
    if total_seconds % 2 == 1:
        activity_index[-1] = 0.55

    # Значения лежат в пределах ~[0.45, 0.65], так что clipping не нужен
    df = pd.DataFrame(
        {
            "sequence_number": seq,
            "people_count": people_count,
            "active_people_count": active_people_count,
            "activity_index": activity_index,
        }
    )
    return df


# seconds_df, который будем использовать везде
SECONDS_DF = build_seconds_df()

# Таблица "что сейчас на экране" — фиксированная, по твоим данным
STATIC_TABLE = pd.DataFrame(
    [
        {"id": 9, "worker_type": "mechanic", "status": "working"},
        {"id": 11, "worker_type": "other", "status": "walking"},
    ],
    columns=["id", "worker_type", "status"],
)


# --------------------
# Рендеринг графиков и текста
# --------------------

def make_main_figure(seconds_df: pd.DataFrame) -> go.Figure:
    """
    График по seconds:
      - people_count
      - active_people_count
      - activity_index
    """
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
    """
    KPI по видео. Параметры синтетических данных подобраны так, чтобы:
      - длительность ≈ 12 минут;
      - максимум работающих людей = 3;
      - минимум = 0;
      - средний индекс активности = 0.55.
    """
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


def train_info_markdown() -> str:
    """
    Статическая плашка о поезде.
    """
    arrival = "2022-03-20 04:35"
    departure = "2022-03-22 04:50"

    md = f"""
### Информация о поезде

- Номер поезда: **{TRAIN_NUMBER}**
- Время прибытия: {arrival}
- Время отправления: {departure}
"""
    return md


# --------------------
# Переключение видео / теплокарты
# --------------------

def switch_media(mode: str):
    """
    'Видео' / 'Размеченное видео' -> gr.Video
    'Тепловая карта'              -> gr.Image
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
# Основной callback: без БД, всё из синтетических данных
# --------------------

async def update_by_time(current_t: float):
    """
    Вызывается каждые 0.5 сек.
    Делает:
      - seconds_df -> график и KPI;
      - статичная таблица (по твоим данным);
      - статичная информация о поезде.
    """
    t = float(current_t or 0.0)

    seconds_df = SECONDS_DF
    table_df = STATIC_TABLE

    fig = make_main_figure(seconds_df)
    kpi_md = kpi_markdown(seconds_df)
    train_md = train_info_markdown()

    # main_plot, table_now, time_state, kpi_box, train_box
    return fig, table_df, t, kpi_md, train_md


# --------------------
# Начальные значения (сразу реальные, а не "загрузка")
# --------------------

INITIAL_FIG = make_main_figure(SECONDS_DF)
INIT_KPI = kpi_markdown(SECONDS_DF)
INIT_TRAIN_MD = train_info_markdown()


# --------------------
# JS: читаем время видео
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
            video = gr.Video(
                value=str(VIDEO_PATH) if VIDEO_PATH.exists() else None,
                format=VIDEO_FORMAT,
                label="Видео",
                elem_id="video_player",
                visible=True,
            )

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
                value=STATIC_TABLE,
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
                    value=INITIAL_FIG,
                )

    # Таймер опрашивает видео на фронте
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
