import cv2
import csv
import math
import numpy as np
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

import asyncio
from datetime import datetime, timedelta
import re

import easyocr  # OCR по естественным сценам

from db_app import (
    AsyncSessionLocal,
    create_tables,
    record_frame_activity,
    Train,
    create_train,
    update_train,
)
from sqlalchemy import select

# ==============================
#       НАСТРОЙКИ ПУТЕЙ
# ==============================

VIDEO_PATH = "data/input/video.mp4"                 # входное видео
OUTPUT_VIDEO_PATH = "data/output/result.mp4"        # видео с боксами, ролями, активностями
OUTPUT_CSV_PATH = "data/output/tracks.csv"           # лог в CSV
OUTPUT_HEATMAP_PATH = "data/output/heatmap.png"      # тепловая карта по всем кадрам

DET_MODEL_PATH = "models/yolo11m.pt"        # detect-модель (person/train)
CLS_MODEL_PATH = "models/yolo11s-cls.pt"    # cls-модель ролей (worker/other/...)
TRACKER_CFG = "models/custom_bytetrack.yaml"  # конфиг трекера

# ==============================
#   ПАРАМЕТРЫ ДЕТЕКТОРА/ТРЕКЕРА
# ==============================

CONF_THRES = 0.4
IOU_THRES = 0.5
IMG_SIZE = 640

# индексы классов в det-модели
PERSON_CLASS_ID = 0
TRAIN_CLASS_ID = 1

# обработка только каждого n-го кадра
FRAME_STRIDE = 1  # 1 = каждый кадр, 3 = каждый третий и т.п.

# ==============================
#   ПАРАМЕТРЫ CLS-МОДЕЛИ РОЛЕЙ
# ==============================

ROLE_CONF_MIN = 0.6         # ниже этого доверия не меняем роль
ROLE_WINDOW = 10            # окно для сглаживания роли
ROLE_RECLASSIFY_EVERY = 15  # как часто дёргать классификатор (в кадрах)

MIN_PERSON_H = 40           # минимальная высота бокса человека для CLS
MIN_PERSON_W = 20           # минимальная ширина бокса человека для CLS

# ==============================
#   ПАРАМЕТРЫ СКОРОСТИ/АКТИВНОСТИ
# ==============================

FPS = 25  # обновится из видео

STILL_MAX_SPEED = 10       # px/сек — стоит/почти не двигается
WALK_SPEED_MIN = 10        # px/сек — начинается ходьба
WALK_SPEED_MAX = 120       # px/сек — быстрые перемещения (на будущее)

MOTION_WINDOW = 5          # окно для скорости
ACTIVITY_WINDOW = 15       # окно для сглаживания активности

# ==============================
#        ЗОНЫ В КАДРЕ
# ==============================

WORK_ZONE = (0, 0, 1000, 1000)
WALK_ZONE = (0, 0, 1000, 1000)

# ==============================
#    ПОЛИГОН ЗОНЫ ПОЕЗДА
# ==============================

TRAIN_POLYGON_NORM = [
    (0.48, 0.22),
    (0.59, 1.0),
    (1.0, 1.0),
    (0.53, 0.22),
]
TRAIN_POLYGON = []  # сюда положим пиксельные координаты после чтения видео

# ==============================
#       ТЕПЛОВАЯ КАРТА
# ==============================

HEATMAP_DOWNSCALE = 4  # 4 → карта ~ в 16 раз меньше по пикселям

# ==============================
#   ПОЕЗД: OCR и логи прибытия
# ==============================

MIN_TRAIN_OCR_W = 80
MIN_TRAIN_OCR_H = 40

TRAIN_ABSENT_GRACE_FRAMES = 10
MIN_TRAIN_PRESENCE_FRAMES = 20  # уточним после чтения FPS

# OCR: EasyOCR + голосование по кадрам
OCR_READER = easyocr.Reader(['ru', 'en'], gpu=True)
MIN_OCR_VOTES = 5

# ==============================
#   СКРИНШОТЫ ПОДОЗРИТЕЛЬНЫХ
# ==============================

SCREENSHOT_DIR = "data/output/alerts"
ALERT_COOLDOWN_SEC = 3.0   # минимум N секунд между скринами


# ==============================
#       ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def ocr_train_number_from_bbox(frame, bbox):
    """
    OCR номера поезда из bbox поезда через EasyOCR.
    Возвращает 'ЭП20-076' / 'ЭП20' / '076' или None.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(0, min(x2, w_frame - 1))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(0, min(y2, h_frame - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 40 or rw < 80:
        return None

    def clean_text(text: str) -> str:
        text = text.upper()
        return re.sub(r"[^0-9A-ZА-Я-]", "", text)

    def run_easyocr(img) -> str:
        try:
            res = OCR_READER.readtext(img, detail=0)
        except Exception as e:
            print(f"[OCR] error: {e}")
            return ""
        if not res:
            return ""
        candidate = max(res, key=len)
        return clean_text(candidate)

    mapping = {
        "3": "Э",
        "З": "Э",
        "E": "Э",
        "Ё": "Э",

        "P": "Р",
        "C": "С",
        "X": "Х",
        "Y": "У",
        "A": "А",
        "B": "В",
        "K": "К",
        "M": "М",
        "H": "Н",
        "O": "О",
    }

    def normalize_series(raw: str) -> str:
        raw = clean_text(raw)
        out = []
        for ch in raw:
            if ch.isdigit():
                out.append(ch)
            else:
                out.append(mapping.get(ch, ch))
        s = "".join(out)
        if s and s[0] in {"3", "З", "E", "Ё"}:
            s = "Э" + s[1:]
        return s

    # цифры справа
    d_x1 = int(rw * 0.55)
    d_x2 = int(rw * 0.98)
    d_y1 = int(rh * 0.55)
    d_y2 = int(rh * 0.98)

    d_x1 = max(0, min(d_x1, rw - 1))
    d_x2 = max(0, min(d_x2, rw))
    d_y1 = max(0, min(d_y1, rh - 1))
    d_y2 = max(0, min(d_y2, rh))

    digits = ""
    if d_x2 > d_x1 and d_y2 > d_y1:
        digits_roi = roi[d_y1:d_y2, d_x1:d_x2]
        if digits_roi.size > 0:
            d_text = run_easyocr(digits_roi)
            digits = re.sub(r"\D", "", d_text)
            if len(digits) > 3:
                digits = digits[-3:]
            if len(digits) < 3:
                digits = ""

    # серия слева
    s_x1 = int(rw * 0.02)
    s_x2 = int(rw * 0.70)
    s_y1 = int(rh * 0.35)
    s_y2 = int(rh * 0.95)

    s_x1 = max(0, min(s_x1, rw - 1))
    s_x2 = max(0, min(s_x2, rw))
    s_y1 = max(0, min(s_y1, rh - 1))
    s_y2 = max(0, min(s_y2, rh))

    series = ""
    if s_x2 > s_x1 and s_y2 > s_y1:
        series_roi = roi[s_y1:s_y2, s_x1:s_x2]
        if series_roi.size > 0:
            s_text = run_easyocr(series_roi)
            series = normalize_series(s_text)
            if len(series) < 2:
                series = ""

    # fallback по всей нижней части
    if not series:
        b_y1 = int(rh * 0.40)
        b_y2 = int(rh * 0.99)
        b_x1 = int(rw * 0.02)
        b_x2 = int(rw * 0.98)

        b_y1 = max(0, min(b_y1, rh - 1))
        b_y2 = max(0, min(b_y2, rh))
        b_x1 = max(0, min(b_x1, rw - 1))
        b_x2 = max(0, min(b_x2, rw))

        if b_x2 > b_x1 and b_y2 > b_y1:
            bottom = roi[b_y1:b_y2, b_x1:b_x2]
            if bottom.size > 0:
                b_text_raw = run_easyocr(bottom)
                b_text = normalize_series(b_text_raw)
                m = re.search(r"[А-Я]{1,3}[0-9]{2}", b_text)
                if m:
                    series = m.group(0)

    if series and digits:
        return f"{series}-{digits}"
    if digits:
        return digits
    if series:
        return series
    return None


async def init_db():
    await create_tables()


async def save_frame_to_db(train_number, people_info, activity_index):
    async with AsyncSessionLocal() as db:
        await record_frame_activity(
            db=db,
            train_number=train_number,
            people_info=people_info,
            activity_index=activity_index,
        )


async def set_train_times_in_db(train_number, arrival_time, departure_time):
    """
    Обновляет времена поезда (по номеру).
    ВАЖНО: обновляет только те поля, которые не None.
    Если поезда не было — создаёт его.
    """
    fields = {}
    if arrival_time is not None:
        fields["arrival_time"] = arrival_time
    if departure_time is not None:
        fields["departure_time"] = departure_time

    if not fields:
        print("[DB][train_times] Нет полей для обновления, выходим")
        return

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Train).where(Train.number == train_number))
        train = result.scalar_one_or_none()

        if train:
            print(f"[DB][train_times] Обновляем поезд id={train.id}, поля={list(fields.keys())}")
            await update_train(db, train_id=train.id, **fields)
        else:
            print(f"[DB][train_times] Поезд не найден, создаём: number={train_number}, поля={list(fields.keys())}")
            await create_train(
                db,
                number=train_number,
                arrival_time=fields.get("arrival_time"),
                departure_time=fields.get("departure_time"),
            )


def point_in_rect(x, y, rect):
    x_min, y_min, x_max, y_max = rect
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def get_zone_label(cx, cy):
    if point_in_rect(cx, cy, WORK_ZONE):
        return "work"
    if point_in_rect(cx, cy, WALK_ZONE):
        return "walk"
    return "other"


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i

    return inside


def classify_person_crop(crop_bgr, cls_model, cls_names):
    result = cls_model(crop_bgr, imgsz=224, verbose=False)[0]
    probs = result.probs
    top_idx = int(probs.top1)
    role_label = cls_names[top_idx]
    conf = float(probs.top1conf)
    return role_label, conf


def update_track_role(state, frame_idx, new_role, new_conf):
    role_state = state.setdefault(
        "role",
        {
            "current_role": "",
            "current_conf": 0.0,
            "last_frame": -ROLE_RECLASSIFY_EVERY,
            "labels_history": [],
            "confs_history": [],
        },
    )

    if new_conf < ROLE_CONF_MIN:
        role_state["last_frame"] = frame_idx
        return role_state

    role_state["last_frame"] = frame_idx

    labels = role_state["labels_history"]
    confs = role_state["confs_history"]

    labels.append(new_role)
    confs.append(new_conf)

    if len(labels) > ROLE_WINDOW:
        labels[:] = labels[-ROLE_WINDOW:]
        confs[:] = confs[-ROLE_WINDOW:]

    counts = Counter(labels)
    most_common_role, _ = counts.most_common(1)[0]
    role_confs = [c for r, c in zip(labels, confs) if r == most_common_role]
    avg_conf = sum(role_confs) / max(1, len(role_confs))

    role_state["current_role"] = most_common_role
    role_state["current_conf"] = avg_conf
    role_state["labels_history"] = labels
    role_state["confs_history"] = confs

    return role_state


def update_track_motion(state, frame_idx, cx, cy, h, fps):
    motion = state.setdefault(
        "motion",
        {
            "positions": [],
            "speed": 0.0,
        },
    )

    positions = motion["positions"]
    positions.append((frame_idx, cx, cy))

    if len(positions) > MOTION_WINDOW:
        positions[:] = positions[-MOTION_WINDOW:]

    if len(positions) < 2:
        motion["speed"] = 0.0
        return

    f0, x0, y0 = positions[0]
    f1, x1, y1 = positions[-1]

    df = max(1, f1 - f0)
    dist = math.hypot(x1 - x0, y1 - y0)
    speed_px_per_sec = dist * (fps / df)
    motion["speed"] = speed_px_per_sec


def infer_activity(zone, speed, in_train_zone):
    if in_train_zone:
        return "working"
    if speed <= STILL_MAX_SPEED:
        return "idle"
    if WALK_SPEED_MIN <= speed <= WALK_SPEED_MAX:
        return "walking"
    return "idle"


def main():
    global WORK_ZONE, WALK_ZONE, FPS, TRAIN_POLYGON, MIN_TRAIN_PRESENCE_FRAMES

    # Инициализируем БД в отдельном event loop
    asyncio.run(init_db())


    current_train_number = None
    ocr_votes = Counter()

    train_present_prev = False
    train_absent_streak = 0
    train_present_duration_frames = 0
    episode_start_frame = None

    train_arrival_time = None
    train_departure_time = None

    last_alert_frame = -10_000_000
    Path(SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    FPS = fps

    cap.release()

    effective_fps_for_presence = fps / FRAME_STRIDE
    MIN_TRAIN_PRESENCE_FRAMES = max(10, int(effective_fps_for_presence * 1.5))

    print(f"Видео: {width}x{height}, FPS={fps:.2f}")
    print(f"FRAME_STRIDE = {FRAME_STRIDE}")
    print(f"MIN_TRAIN_PRESENCE_FRAMES = {MIN_TRAIN_PRESENCE_FRAMES}")

    WALK_ZONE = (0, 0, int(width * 0.48), height)
    WORK_ZONE = (int(width * 0.48), 0, width, height)

    TRAIN_POLYGON[:] = [
        (int(x * width), int(y * height)) for (x, y) in TRAIN_POLYGON_NORM
    ]

    print(f"WORK_ZONE: {WORK_ZONE}")
    print(f"WALK_ZONE: {WALK_ZONE}")
    print(f"TRAIN_POLYGON: {TRAIN_POLYGON}")

    heatmap_h = height // HEATMAP_DOWNSCALE
    heatmap_w = width // HEATMAP_DOWNSCALE
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

    det_model = YOLO(DET_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)
    cls_names = cls_model.model.names

    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame",
            "track_id",
            "det_class_id",
            "det_class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "role",
            "role_conf",
            "speed_px_per_sec",
            "zone",
            "in_train_polygon",
            "activity",
        ]
    )

    track_states = {}

    results_gen = det_model.track(
        source=VIDEO_PATH,
        stream=True,
        tracker=TRACKER_CFG,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        classes=[PERSON_CLASS_ID, TRAIN_CLASS_ID],
        verbose=False,
        persist=True,
    )

    print("Старт обработки...")
    last_frame_idx = -1

    for frame_idx, result in enumerate(results_gen):
        last_frame_idx = frame_idx
        frame = result.orig_img

        if frame_idx % FRAME_STRIDE != 0:
            out_writer.write(frame)
            continue

        boxes = result.boxes

        frame_people = []
        total_persons = 0
        working_persons = 0

        train_detected_raw = False

        if boxes is None or len(boxes) == 0:
            xyxy = np.empty((0, 4), dtype=float)
            cls_arr = np.empty((0,), dtype=float)
            ids = []
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy()
            ids = boxes.id
            if ids is not None:
                ids = ids.cpu().numpy()
            else:
                ids = [-1] * len(xyxy)

        h_frame, w_frame = frame.shape[:2]

        for bbox, cls_id, track_id in zip(xyxy, cls_arr, ids):
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1

            class_id = int(cls_id)
            if class_id == PERSON_CLASS_ID:
                det_class_name = "person"
            elif class_id == TRAIN_CLASS_ID:
                det_class_name = "train"
            else:
                det_class_name = f"class_{class_id}"

            # поезд
            if det_class_name == "train":
                train_detected_raw = True

                if (
                    current_train_number is None
                    and w >= MIN_TRAIN_OCR_W
                    and h >= MIN_TRAIN_OCR_H
                ):
                    candidate = ocr_train_number_from_bbox(frame, (x1, y1, x2, y2))
                    if candidate:
                        ocr_votes[candidate] += 1
                        best_candidate, best_votes = ocr_votes.most_common(1)[0]
                        if best_votes >= MIN_OCR_VOTES and current_train_number is None:
                            current_train_number = best_candidate
                            print(
                                f"[OCR] Номер поезда принят по голосованию: "
                                f"{current_train_number} (кадров: {best_votes})"
                            )
                            if (
                                episode_start_frame is not None
                                and train_arrival_time is None
                            ):
                                train_arrival_time = datetime.utcnow()
                                print(
                                    f"[TRAIN] Прибытие поезда (OCR позже, PC time): "
                                    f"{train_arrival_time}"
                                )

            current_role = ""
            role_conf = 0.0
            speed = 0.0
            zone = ""
            current_activity = ""
            in_train_poly = False

            state = None
            if track_id != -1:
                state = track_states.get(track_id)
                if state is None:
                    state = {}
                    track_states[track_id] = state

            if det_class_name == "person" and state is not None:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                update_track_motion(state, frame_idx, cx, cy, h, fps)
                motion_state = state.get("motion", {})
                speed = motion_state.get("speed", 0.0)

                role_state = state.setdefault(
                    "role",
                    {
                        "current_role": "",
                        "current_conf": 0.0,
                        "last_frame": -ROLE_RECLASSIFY_EVERY,
                        "labels_history": [],
                        "confs_history": [],
                    },
                )

                need_role_cls = False
                if h >= MIN_PERSON_H and w >= MIN_PERSON_W:
                    if frame_idx - role_state["last_frame"] >= ROLE_RECLASSIFY_EVERY:
                        need_role_cls = True
                    if not role_state["current_role"]:
                        need_role_cls = True

                if need_role_cls:
                    x1c = max(0, min(x1, w_frame - 1))
                    x2c = max(0, min(x2, w_frame - 1))
                    y1c = max(0, min(y1, h_frame - 1))
                    y2c = max(0, min(y2, h_frame - 1))
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        role_label, conf = classify_person_crop(
                            crop, cls_model, cls_names
                        )
                        role_state = update_track_role(
                            state, frame_idx, role_label, conf
                        )

                current_role = role_state["current_role"]
                role_conf = role_state["current_conf"]

                zone = get_zone_label(cx, cy)
                in_train_poly = point_in_polygon(cx, cy, TRAIN_POLYGON)

                activity_state = state.setdefault(
                    "activity",
                    {
                        "current_activity": "",
                        "history": [],
                    },
                )
                raw_activity = infer_activity(zone, speed, in_train_poly)
                hist = activity_state["history"]
                hist.append(raw_activity)
                if len(hist) > ACTIVITY_WINDOW:
                    hist[:] = hist[-ACTIVITY_WINDOW:]
                counts = Counter(hist)
                current_activity, _ = counts.most_common(1)[0]
                activity_state["current_activity"] = current_activity

                # триггер скриншота
                if zone == "work" and current_role and current_role.lower() == "other":
                    if (frame_idx - last_alert_frame) >= ALERT_COOLDOWN_SEC * fps:
                        last_alert_frame = frame_idx
                        sec_from_start = frame_idx / fps if fps else 0.0
                        time_delta = timedelta(seconds=sec_from_start)
                        time_str = str(time_delta)
                        safe_time_str = time_str.replace(":", "-").replace(".", "-")
                        filename = f"{safe_time_str}.jpg"
                        save_path = Path(SCREENSHOT_DIR) / filename
                        cv2.imwrite(str(save_path), frame)
                        print(
                            f"[ALERT] OTHER в рабочей зоне у поезда: "
                            f"frame={frame_idx}, t={sec_from_start:.2f} c → {save_path}"
                        )

                total_persons += 1
                if current_activity == "working":
                    working_persons += 1

                status = "active" if current_activity == "working" else "idle"
                frame_people.append(
                    {
                        "worker_type": current_role or "unknown",
                        "status": status,
                    }
                )

                if current_activity == "working":
                    hx = int(cx / HEATMAP_DOWNSCALE)
                    hy = int(cy / HEATMAP_DOWNSCALE)
                    if 0 <= hx < heatmap_w and 0 <= hy < heatmap_h:
                        heatmap[hy, hx] += 1.0

            # отрисовка
            if det_class_name == "person":
                if current_activity == "working":
                    color = (0, 255, 0)
                elif current_activity == "walking":
                    color = (255, 255, 0)
                elif current_activity == "idle":
                    color = (0, 215, 255)
                else:
                    color = (0, 255, 255)
            elif det_class_name == "train":
                color = (0, 0, 255)
            else:
                color = (255, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_parts = [det_class_name]
            if track_id != -1:
                label_parts.append(f"ID {int(track_id)}")
            if current_role:
                label_parts.append(current_role)
            if current_activity:
                label_parts.append(current_activity)

            label_text = " | ".join(label_parts)
            cv2.putText(
                frame,
                label_text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            csv_writer.writerow(
                [
                    frame_idx,
                    int(track_id) if track_id != -1 else -1,
                    class_id,
                    det_class_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    current_role,
                    f"{role_conf:.3f}",
                    f"{speed:.1f}",
                    zone,
                    int(in_train_poly),
                    current_activity,
                ]
            )

        # логика присутствия поезда
        if train_detected_raw:
            train_absent_streak = 0
        else:
            train_absent_streak += 1

        if train_detected_raw or (
            train_present_prev and train_absent_streak <= TRAIN_ABSENT_GRACE_FRAMES
        ):
            train_present_this = True
        else:
            train_present_this = False

        if train_present_this and not train_present_prev:
            episode_start_frame = frame_idx
            train_present_duration_frames = 1
            print(f"[TRAIN] Поезд появился в кадре: frame={frame_idx}")
        elif train_present_this and train_present_prev:
            train_present_duration_frames += 1

        if (
            train_present_this
            and current_train_number is not None
            and train_arrival_time is None
            and episode_start_frame is not None
            and train_present_duration_frames >= MIN_TRAIN_PRESENCE_FRAMES
        ):
            train_arrival_time = datetime.utcnow()
            print(
                f"[TRAIN] Прибытие поезда (PC time): {train_arrival_time}"
            )

        if (not train_present_this) and train_present_prev:
            if (
                train_present_duration_frames >= MIN_TRAIN_PRESENCE_FRAMES
                and current_train_number is not None
            ):
                train_departure_time = datetime.utcnow()
                print(
                    f"[TRAIN] Отбытие поезда (PC time): {train_departure_time}"
                )
            else:
                print(
                    f"[TRAIN] Короткий шум поезда, игнорируем эпизод "
                    f"(duration={train_present_duration_frames})"
                )

            train_present_duration_frames = 0
            episode_start_frame = None

        train_present_prev = train_present_this

        if current_train_number is not None and total_persons > 0:
            activity_index = (
                working_persons / total_persons if total_persons > 0 else 0.0
            )
            try:
                loop.run_until_complete(
                    save_frame_to_db(
                        current_train_number,
                        frame_people,
                        activity_index,
                    )
                )
            except Exception as e:
                print(f"[DB] Ошибка записи кадра в БД: {e}")

        out_writer.write(frame)

    print("Обработка видео завершена.")

    out_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # финализация departure, если поезд остался "в кадре" к концу видео
    if (
        current_train_number
        and train_arrival_time
        and not train_departure_time
        and last_frame_idx >= 0
    ):
        train_departure_time = datetime.utcnow()
        print(
            f"[TRAIN] Отбытие поезда по концу видео (PC time): {train_departure_time}"
        )

    # отправляем времена поезда в БД (arrival и/или departure)
    if current_train_number:
        try:
            print(
                f"[DB] Пишем времена поезда {current_train_number}: "
                f"arrival={train_arrival_time}, departure={train_departure_time}"
            )
            loop.run_until_complete(
                set_train_times_in_db(
                    current_train_number,
                    train_arrival_time,
                    train_departure_time,
                )
            )
        except Exception as e:
            print(f"[DB] Ошибка обновления времён поезда: {e}")
    else:
        print("[TRAIN] Номер поезда не распознан, времена не записаны в БД.")

    if np.max(heatmap) > 0:
        heat_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heat_uint8 = heat_norm.astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color_resized = cv2.resize(
            heat_color, (width, height), interpolation=cv2.INTER_LINEAR
        )
        Path(OUTPUT_HEATMAP_PATH).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(OUTPUT_HEATMAP_PATH, heat_color_resized)
        print(f"Тепловая карта сохранена в: {OUTPUT_HEATMAP_PATH}")
    else:
        print("Тепловая карта пустая (никто не был в состоянии 'working').")

    print(f"Видео сохранено в: {OUTPUT_VIDEO_PATH}")
    print(f"CSV лог:          {OUTPUT_CSV_PATH}")

    loop.close()


if __name__ == "__main__":
    main()
