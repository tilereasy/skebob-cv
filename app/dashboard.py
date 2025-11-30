import os
import asyncio

import gradio as gr
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqlalchemy import select

from db_app import (
    AsyncSessionLocal,
    Train,
    Second,
    SecondsPeople,
    People,
)

# ---------------------------------------
# ПУТИ К ФАЙЛАМ
# ---------------------------------------
RAW_VIDEO_PATH = "data/input/video.mp4"
MARKED_VIDEO_PATH = "data/output/result.mp4"
HEATMAP_PATH = "data/output/heatmap.png"
TRACKS_CSV_PATH = "data/output/tracks.csv"
ALERTS_DIR = "data/output/alerts"


# =======================================
#      ASYNC УТИЛИТЫ РАБОТЫ С БД
# =======================================

async def _fetch_trains():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Train))
        trains = result.scalars().all()
        choices = []
        for t in trains:
            arr = t.arrival_time.strftime("%Y-%m-%d %H:%M:%S") if t.arrival_time else "—"
            dep = t.departure_time.strftime("%Y-%m-%d %H:%M:%S") if t.departure_time else "—"
            label = f"{t.id}: {t.number} (arr: {arr}, dep: {dep})"
            choices.append(label)
        return choices


async def _fetch_train_seconds_df(train_id: int) -> pd.DataFrame:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Second).where(Second.train_id == train_id).order_by(Second.sequence_number)
        )
        seconds = result.scalars().all()

        rows = []
        for s in seconds:
            rows.append(
                {
                    "seq": s.sequence_number,
                    "timestamp": s.timestamp,
                    "people_count": s.people_count,
                    "active_people": s.active_people_count,
                    "activity_index": s.activity_index,
                }
            )
        return pd.DataFrame(rows)


async def _get_second_id_by_seq(train_id: int, seq: int):
    if seq is None:
        return None
    seq = int(seq)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Second).where(
                Second.train_id == train_id,
                Second.sequence_number == seq,
            )
        )
        second = result.scalar_one_or_none()
        return second.id if second else None


async def _fetch_people_for_second(second_id: int) -> pd.DataFrame:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(SecondsPeople).where(SecondsPeople.second_id == second_id)
        )
        links = result.scalars().all()

        rows = []
        for link in links:
            person = await db.get(People, link.person_id)
            rows.append(
                {
                    "worker_type": person.worker_type if person else "unknown",
                    "status": link.status,
                }
            )

        return pd.DataFrame(rows)


async def _get_train_by_id(train_id: int):
    async with AsyncSessionLocal() as db:
        return await db.get(Train, train_id)


# =======================================
#   ОБЁРТКИ ДЛЯ ВЫЗОВА ИЗ GRADIO (sync)
# =======================================


def load_trains():
    try:
        return asyncio.run(_fetch_trains())
    except Exception as exc:
        print(f"[dashboard] Не удалось загрузить список поездов: {exc}")
        return []


def parse_train_choice(choice: str):
    if not choice:
        return None, None
    try:
        train_id = int(choice.split(":")[0])
    except (ValueError, AttributeError):
        return None, None

    try:
        train = asyncio.run(_get_train_by_id(train_id))
    except Exception as exc:
        print(f"[dashboard] Не удалось получить поезд {train_id}: {exc}")
        train = None

    train_number = train.number if train else "UNKNOWN"
    return train_id, train_number


def load_people_df_for_seq(current_seq, train_id):
    empty = pd.DataFrame(columns=["worker_type", "status"])
    if not train_id or current_seq is None:
        return empty

    try:
        second_id = asyncio.run(_get_second_id_by_seq(train_id, current_seq))
    except Exception as exc:
        print(f"[dashboard] Не удалось получить Second для seq={current_seq}: {exc}")
        return empty

    if not second_id:
        return empty

    try:
        df = asyncio.run(_fetch_people_for_second(second_id))
    except Exception as exc:
        print(f"[dashboard] Не удалось загрузить людей для second_id={second_id}: {exc}")
        return empty

    if df.empty:
        return empty
    return df


# =======================================
#    tracks.csv: ЗАГРУЗКА И СТАТИСТИКА
# =======================================


def load_tracks_for_train(train_number: str):
    if not train_number or not os.path.exists(TRACKS_CSV_PATH):
        return None

    try:
        df = pd.read_csv(TRACKS_CSV_PATH)
    except Exception as exc:
        print(f"[dashboard] Ошибка чтения tracks.csv: {exc}")
        return None

    if "train_number" in df.columns:
        df = df[df["train_number"] == train_number]

    if df.empty:
        return None

    return df


def build_csv_stats_markdown(tracks_df: pd.DataFrame) -> str:
    if tracks_df is None:
        return "_tracks.csv не найден или не содержит данных для этого поезда._"

    md_lines = ["### Доп. статистика по tracks.csv"]

    for col, title in [
        ("people_count", "Люди в кадре (по tracks.csv)"),
        ("active_people", "Работающие люди (по tracks.csv)"),
        ("activity_index", "Индекс активности (по tracks.csv)"),
    ]:
        if col in tracks_df.columns:
            md_lines.append(
                f"- **{title}:** среднее = {tracks_df[col].mean():.2f}, "
                f"максимум = {tracks_df[col].max():.2f}"
            )

    num_cols = tracks_df.select_dtypes(include="number").columns.tolist()
    extra_cols = [
        c for c in num_cols if c not in ["people_count", "active_people", "activity_index"]
    ]

    if extra_cols:
        md_lines.append("")
        md_lines.append("Доп. числовые метрики:")
        for c in extra_cols:
            md_lines.append(
                f"- **{c}:** среднее = {tracks_df[c].mean():.2f}, "
                f"максимум = {tracks_df[c].max():.2f}"
            )

    return "\n".join(md_lines)


# =======================================
#     KPI, ОПАСНЫЕ МОМЕНТЫ, ГРАФИК
# =======================================


def build_kpi_markdown(df_seconds: pd.DataFrame) -> str:
    if df_seconds is None or df_seconds.empty:
        return "Нет данных по выбранному поезду."

    total_seconds = len(df_seconds)
    avg_people = df_seconds["people_count"].mean()
    avg_active = df_seconds["active_people"].mean()
    avg_activity = df_seconds["activity_index"].mean()

    max_people = df_seconds["people_count"].max()
    max_activity = df_seconds["activity_index"].max()

    busiest_row = df_seconds.sort_values("activity_index", ascending=False).iloc[0]
    busiest_seq = int(busiest_row["seq"])

    md = f"""
### KPI по видео (из БД)

- **Длительность наблюдения:** {total_seconds} сек
- **Среднее число людей в кадре:** {avg_people:.2f}
- **Среднее число работающих людей:** {avg_active:.2f}
- **Средний индекс активности:** {avg_activity:.3f}

- **Максимум людей в кадре:** {max_people}
- **Пик активности:** {max_activity:.3f} (на секунде {busiest_seq})
"""
    return md


def build_danger_highlights(train_number: str, df_seconds: pd.DataFrame, top_n: int = 4):
    if df_seconds is None or df_seconds.empty:
        return [], pd.DataFrame()

    top = df_seconds.sort_values("activity_index", ascending=False).head(top_n)

    gallery_items = []
    meta_rows = []

    for _, row in top.iterrows():
        seq = int(row["seq"])
        ts = row["timestamp"]
        pc = int(row["people_count"])
        ac = int(row["active_people"])
        idx = float(row["activity_index"])

        candidates = [
            os.path.join(ALERTS_DIR, f"{train_number}_sec{seq}.jpg"),
            os.path.join(ALERTS_DIR, f"{train_number}_{seq}.jpg"),
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)

        caption = f"sec {seq} | people={pc}, active={ac}, idx={idx:.2f}"
        if img_path:
            gallery_items.append((img_path, caption))

        meta_rows.append(
            {
                "sequence": seq,
                "timestamp": ts,
                "people_count": pc,
                "active_people": ac,
                "activity_index": idx,
            }
        )

    df_meta = pd.DataFrame(meta_rows)
    return gallery_items, df_meta


def build_activity_plot(df_seconds: pd.DataFrame, current_seq):
    fig, ax = plt.subplots(figsize=(6, 3))

    if df_seconds is None or df_seconds.empty:
        ax.set_title("Нет данных для графика")
        fig.tight_layout()
        return fig

    x = df_seconds["seq"]
    ax.plot(x, df_seconds["people_count"], label="Люди в кадре")
    ax.plot(x, df_seconds["active_people"], label="Работающие люди")
    ax.plot(x, df_seconds["activity_index"], label="Индекс активности")

    if current_seq is not None:
        ax.axvline(int(current_seq), linestyle="--", color="black", alpha=0.7)

    ax.set_xlabel("Секунда (sequence_number)")
    ax.set_ylabel("Значение")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Динамика людей и активности")

    fig.tight_layout()
    return fig


def build_train_info_markdown(train_id):
    if not train_id:
        return "Поезд не выбран."

    try:
        train = asyncio.run(_get_train_by_id(train_id))
    except Exception as exc:
        return f"Ошибка загрузки поезда: {exc}"

    if not train:
        return "Поезд не найден в БД."

    arr = train.arrival_time.strftime("%Y-%m-%d %H:%M:%S") if train.arrival_time else "—"
    dep = train.departure_time.strftime("%Y-%m-%d %H:%M:%S") if train.departure_time else "—"

    return f"""
### Информация о поезде

- **Номер:** {train.number}
- **Время прибытия:** {arr}
- **Время отбытия:** {dep}
"""


# =======================================
#          CALLBACK-ФУНКЦИИ UI
# =======================================


def on_train_change(train_choice: str):
    train_id, train_number = parse_train_choice(train_choice)

    slider_disabled = gr.Slider.update(minimum=0, maximum=0, value=0, interactive=False)
    empty_df = pd.DataFrame()

    if not train_id:
        csv_md = build_csv_stats_markdown(load_tracks_for_train(train_number))
        kpi_md = "Поезд не выбран.\n\n" + csv_md
        return (
            slider_disabled,
            kpi_md,
            [],
            empty_df,
            build_activity_plot(empty_df, None),
            pd.DataFrame(columns=["worker_type", "status"]),
            build_train_info_markdown(None),
            empty_df,
            None,
            None,
        )

    try:
        df_seconds = asyncio.run(_fetch_train_seconds_df(train_id))
    except Exception as exc:
        print(f"[dashboard] Ошибка загрузки секунд по поезду {train_id}: {exc}")
        df_seconds = pd.DataFrame()

    if df_seconds.empty:
        fig = build_activity_plot(df_seconds, None)
        base_kpi = "Нет данных по секундам для этого поезда."
        tracks_df = load_tracks_for_train(train_number)
        csv_kpi = build_csv_stats_markdown(tracks_df)
        kpi_md = base_kpi + "\n\n" + csv_kpi

        gallery_items, danger_df = [], pd.DataFrame()
        people_now = pd.DataFrame(columns=["worker_type", "status"])
        train_info = build_train_info_markdown(train_id)

        return (
            slider_disabled,
            kpi_md,
            gallery_items,
            danger_df,
            fig,
            people_now,
            train_info,
            df_seconds,
            train_id,
            train_number,
        )

    min_seq = int(df_seconds["seq"].min())
    max_seq = int(df_seconds["seq"].max())
    current_seq = min_seq

    slider_update = gr.Slider.update(
        minimum=min_seq,
        maximum=max_seq,
        value=current_seq,
        step=1,
        interactive=True,
    )

    base_kpi = build_kpi_markdown(df_seconds)
    tracks_df = load_tracks_for_train(train_number)
    csv_kpi = build_csv_stats_markdown(tracks_df)
    kpi_md = base_kpi + "\n\n" + csv_kpi

    gallery_items, danger_df = build_danger_highlights(train_number, df_seconds)
    fig = build_activity_plot(df_seconds, current_seq)

    people_now = load_people_df_for_seq(current_seq, train_id)
    train_info = build_train_info_markdown(train_id)

    return (
        slider_update,
        kpi_md,
        gallery_items,
        danger_df,
        fig,
        people_now,
        train_info,
        df_seconds,
        train_id,
        train_number,
    )


def on_second_change(current_seq, seconds_df: pd.DataFrame, train_id: int):
    if seconds_df is None or seconds_df.empty or not train_id:
        empty_df = pd.DataFrame(columns=["worker_type", "status"])
        fig = build_activity_plot(pd.DataFrame(), None)
        return empty_df, fig

    people_now = load_people_df_for_seq(current_seq, train_id)
    fig = build_activity_plot(seconds_df, current_seq)
    return people_now, fig


def on_mode_change(mode: str):
    if mode == "Сырое видео":
        return (
            gr.Video.update(
                value=RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else None,
                visible=True,
            ),
            gr.Video.update(visible=False),
            gr.Image.update(visible=False),
        )
    elif mode == "Размеченное видео":
        return (
            gr.Video.update(visible=False),
            gr.Video.update(
                value=MARKED_VIDEO_PATH if os.path.exists(MARKED_VIDEO_PATH) else None,
                visible=True,
            ),
            gr.Image.update(visible=False),
        )
    else:  # Тепловая карта
        return (
            gr.Video.update(visible=False),
            gr.Video.update(visible=False),
            gr.Image.update(
                value=HEATMAP_PATH if os.path.exists(HEATMAP_PATH) else None,
                visible=True,
            ),
        )


# =======================================
#              UI НА GRADIO
# =======================================


with gr.Blocks(title="Depo Safety Dashboard") as demo:
    gr.Markdown("# 🚆 Depo Safety Dashboard")

    seconds_state = gr.State()
    train_id_state = gr.State()
    train_number_state = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            train_choices = load_trains()
            train_dropdown = gr.Dropdown(
                label="Поезд",
                choices=train_choices,
                value=train_choices[0] if train_choices else None,
                interactive=True,
            )

            with gr.Row():
                raw_video = gr.Video(
                    label="Видео",
                    value=RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else None,
                    visible=True,
                )
                marked_video = gr.Video(
                    label="Размеченное видео",
                    value=MARKED_VIDEO_PATH if os.path.exists(MARKED_VIDEO_PATH) else None,
                    visible=False,
                )
                heatmap_image = gr.Image(
                    label="Тепловая карта",
                    value=HEATMAP_PATH if os.path.exists(HEATMAP_PATH) else None,
                    visible=False,
                    type="filepath",
                )

            mode_radio = gr.Radio(
                ["Сырое видео", "Размеченное видео", "Тепловая карта"],
                label="Режим просмотра",
                value="Сырое видео",
                interactive=True,
            )

            current_second = gr.Slider(
                label="Текущая секунда (sequence_number)",
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                interactive=False,
            )

            gr.Markdown("### Кто сейчас в кадре")
            people_now_table = gr.DataFrame(
                headers=["worker_type", "status"],
                value=pd.DataFrame(columns=["worker_type", "status"]),
                interactive=False,
            )

        with gr.Column(scale=2):
            train_info_md = gr.Markdown("Информация о поезде появится здесь")
            kpi_md = gr.Markdown("KPI появятся после выбора поезда")

            gr.Markdown("### Опасные моменты (highlights)")
            danger_gallery = gr.Gallery(
                label="Danger highlights (картинки, если есть)",
                show_label=True,
                columns=2,
                height=200,
            )
            danger_table = gr.DataFrame(
                label="Список опасных секунд",
                interactive=False,
            )

            gr.Markdown("### График людей и активности")
            activity_plot = gr.Plot()

    train_dropdown.change(
        fn=on_train_change,
        inputs=train_dropdown,
        outputs=[
            current_second,
            kpi_md,
            danger_gallery,
            danger_table,
            activity_plot,
            people_now_table,
            train_info_md,
            seconds_state,
            train_id_state,
            train_number_state,
        ],
    )

    current_second.change(
        fn=on_second_change,
        inputs=[current_second, seconds_state, train_id_state],
        outputs=[people_now_table, activity_plot],
    )

    mode_radio.change(
        fn=on_mode_change,
        inputs=mode_radio,
        outputs=[raw_video, marked_video, heatmap_image],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
