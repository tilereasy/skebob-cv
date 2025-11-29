import os
from pathlib import Path
import math
import random
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import gradio as gr


# --------------------
# Пути и окружение
# --------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "output"

# Имя файла видео
VIDEO_FILENAME = "result.mov"
VIDEO_PATH = VIDEO_DIR / VIDEO_FILENAME

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


# --------------------
# Тестовые данные (имитация структуры БД)
# --------------------

def create_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Имитация данных согласно схеме БД:

        train          seconds              people           seconds_people
        -----          -------              ------           --------------
        id             id                   id               id
        number         track_id             worker_type      person_id
        arrival_time   people_count                           second_id
        departure_time active_people_count                    status
                       activity_index
                       train_id

    Здесь:
    - seconds.id интерпретируется как "номер секунды от начала видео" (t_sec).
    - seconds_people задаёт M:N связь между people и seconds + статус в каждую секунду.

    Возвращает
    ----------
    train_df : pd.DataFrame
    seconds_df : pd.DataFrame
    people_df : pd.DataFrame
    seconds_people_df : pd.DataFrame
    """
    random.seed(42)

    # ---- train ----
    train_rows = [
        {
            "id": 1,
            "number": "A123",
            "arrival_time": pd.Timestamp("2023-01-01 10:00:00"),
            "departure_time": pd.Timestamp("2023-01-01 10:30:00"),
        }
    ]
    train_df = pd.DataFrame(train_rows)

    # Длительность видео в секундах (пример: 30 минут).
    total_seconds = 30 * 60
    seconds_range = range(total_seconds)

    # ---- seconds ----
    seconds_rows = []
    for t in seconds_range:
        # Базовый уровень количества людей с плавной волатильностью.
        # Используем периодическую функцию по времени (t ~ секунды).
        base = 3 + int(2 * math.sin(t / 60.0))

        people_present = max(0, base + random.randint(-1, 2))

        if people_present == 0:
            active = 0
        else:
            active = random.randint(0, people_present)

        if people_present > 0:
            ai = (active / people_present) + random.uniform(-0.1, 0.1)
        else:
            ai = 0.0
        activity_index = max(0.0, min(1.0, ai))

        seconds_rows.append(
            {
                "id": t,  # номер секунды от начала ролика
                "track_id": 1,  # фиктивный идентификатор трека/камеры
                "people_count": people_present,
                "active_people_count": active,
                "activity_index": activity_index,
                "train_id": 1,
            }
        )

    seconds_df = pd.DataFrame(seconds_rows)

    # ---- people ----
    people_rows = []
    seconds_people_rows = []

    statuses = ["idle", "walking", "working", "inspecting"]

    for pid in range(1, 6):
        worker_type = "worker" if pid <= 3 else "mechanic"
        people_rows.append(
            {
                "id": pid,
                "worker_type": worker_type,
            }
        )

        # Интервал присутствия персоны.
        first_seen = random.randint(0, total_seconds // 3)
        last_seen = first_seen + random.randint(60, total_seconds // 2)
        last_seen = min(last_seen, total_seconds - 1)

        # Разбиваем интервал на эпизоды с разными статусами.
        cur = first_seen
        while cur <= last_seen:
            dur = random.randint(10, 60)
            end = min(cur + dur, last_seen + 1)  # end не включительно
            status = random.choice(statuses)

            # Для каждой секунды эпизода создаём запись в seconds_people.
            for t in range(cur, end):
                seconds_people_rows.append(
                    {
                        "person_id": pid,
                        "second_id": t,  # ссылка на seconds.id
                        "status": status,
                    }
                )
            cur = end

    people_df = pd.DataFrame(people_rows)

    seconds_people_df = pd.DataFrame(seconds_people_rows)
    # Вводим surrogate PK для seconds_people.
    seconds_people_df.insert(0, "id", range(1, len(seconds_people_df) + 1))

    return train_df, seconds_df, people_df, seconds_people_df


TRAIN_DF, SECONDS_DF, PEOPLE_DF, SECONDS_PEOPLE_DF = create_test_data()


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

    fig = go.Figure()

    # Линия "людей в кадре".
    fig.add_trace(
        go.Scatter(
            x=seconds_df["id"],
            y=seconds_df["people_count"],
            mode="lines",
            name="Людей в кадре",
            line=dict(color="#1f77b4"),
        )
    )

    # Линия "работающих людей".
    fig.add_trace(
        go.Scatter(
            x=seconds_df["id"],
            y=seconds_df["active_people_count"],
            mode="lines",
            name="Работающих людей",
            line=dict(color="#2ca02c"),
        )
    )

    # Индекс активности выводим на вторую ось Y.
    fig.add_trace(
        go.Scatter(
            x=seconds_df["id"],
            y=seconds_df["activity_index"],
            mode="lines",
            name="Индекс активности",
            yaxis="y2",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )

    fig.update_layout(
        title="Количество людей, работающих людей и индекс активности",
        xaxis=dict(title="Время, сек (seconds.id)"),
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

    Требуемый формат:
        - id человека
        - его специальность (worker_type)
        - его статус в данный момент (status)

    Логика:
        1. Преобразуем t_sec к целой секунде (floor).
        2. Находим строки seconds_people для second_id == выбранной секунде.
        3. Джойним people, чтобы получить worker_type.
        4. Возвращаем таблицу c колонками [id, worker_type, status].
    """
    if people_df.empty or seconds_df.empty or seconds_people_df.empty:
        return pd.DataFrame(columns=["id", "worker_type", "status"])

    # Ограничиваем секунду диапазоном доступных значений seconds.id.
    sec_min = int(seconds_df["id"].min())
    sec_max = int(seconds_df["id"].max())
    sec_id = int(t_sec)
    sec_id = max(sec_min, min(sec_id, sec_max))

    sub = seconds_people_df[seconds_people_df["second_id"] == sec_id]
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
    Формирование блока KPI по видео.

    Требования:
        - длительность видео;
        - максимум и минимум работающих людей в кадре за всё видео;
        - средний индекс активности.
    """
    if seconds_df.empty:
        return "Нет данных"

    total_seconds = seconds_df["id"].nunique()

    duration_min = total_seconds // 60
    duration_sec = total_seconds % 60

    max_active = seconds_df["active_people_count"].max()
    min_active = seconds_df["active_people_count"].min()
    avg_activity = seconds_df["activity_index"].mean()

    md = f"""
**KPI по видео**

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

    Параметры
    ---------
    current_t : float
        Текущее время воспроизведения видео (секунды),
        получаемое с фронтенда через JS (video.currentTime).
    """
    t = float(current_t or 0.0)

    fig = make_main_figure(SECONDS_DF)
    tbl = table_at_time(PEOPLE_DF, SECONDS_DF, SECONDS_PEOPLE_DF, t)

    return fig, tbl, t


def mode_to_video_value(mode: str):
    """
    Маппинг выбранного режима воспроизведения (radio) на путь к видеофайлу.

    Возвращает:
        - строку с путём (для gr.Video), если файл существует;
        - None, если файл отсутствует.
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

# JS-функция для получения текущего времени видео.
#
# Вызов:
#   - дергается таймером (gr.Timer) каждые N секунд;
#   - возвращает video.currentTime (секунды с плавающей точностью);
#   - результат записывается в скрытый компонент current_time.
READ_VIDEO_TIME_JS = """
() => {
  let video = null;

  const root = document.querySelector('#video_player');
  if (root) {
    const tag = (root.tagName || '').toLowerCase();
    if (tag === 'video') {
      // elem_id висит прямо на теге <video>
      video = root;
    } else {
      // elem_id висит на обёртке, ищем <video> внутри
      video = root.querySelector('video');
    }
  }

  // Fallback: первый <video> на странице.
  if (!video) {
    video = document.querySelector('video');
  }

  if (!video) {
    // Если по какой-то причине видео так и не нашли – возвращаем 0.
    return 0;
  }

  // Текущее время воспроизведения (в секундах).
  return video.currentTime || 0;
}
"""


# --------------------
# UI
# --------------------

with gr.Blocks() as demo:
    # Состояние с текущим временем на стороне сервера (используется Python-логикой).
    time_state = gr.State(0.0)

    # Текущее время видео, синхронизируемое с фронтендом (через JS + Timer).
    # Компонент скрыт из UI, используется только как транспорт.
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
    # allowed_paths ограничивает доступ Gradio к файловой системе.
    demo.launch(debug=True, allowed_paths=[str(VIDEO_DIR)])
