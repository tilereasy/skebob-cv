import cv2
import csv
import math
import numpy as np
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# ==============================
#       НАСТРОЙКИ ПУТЕЙ
# ==============================

VIDEO_PATH = "data/input/short_video.mp4"                 # входное видео
OUTPUT_VIDEO_PATH = "data/output/depo_tracked.mp4"  # видео с боксами, ролями, активностями
OUTPUT_CSV_PATH = "data/output/depo_tracks.csv"     # лог в CSV
OUTPUT_HEATMAP_PATH = "data/output/depo_heatmap.png"  # тепловая карта по всем кадрам

DET_MODEL_PATH = "models/yolo11m.pt"        # detect-модель (person/train)
CLS_MODEL_PATH = "models/yolo11s-cls.pt"    # cls-модель ролей (worker/other/...)
TRACKER_CFG = "models/custom_bytetrack.yaml"  # конфиг трекера

# ==============================
#   ПАРАМЕТРЫ ДЕТЕКТОРА/ТРЕКЕРА
# ==============================

CONF_THRES = 0.4
IOU_THRES = 0.5
IMG_SIZE = 960

# подгони под свои ids в det-модели
PERSON_CLASS_ID = 0
TRAIN_CLASS_ID = 1

# ==============================
#    ПАРАМЕТРЫ РОЛЕЙ (CLS-МОДЕЛЬ)
# ==============================

ROLE_CONF_MIN = 0.6        # ниже этого доверия не меняем роль
ROLE_WINDOW = 10           # окно для сглаживания роли
ROLE_RECLASSIFY_EVERY = 15 # как часто дёргать классификатор (в кадрах)

MIN_PERSON_H = 60          # минимальная высота бокса для роли
MIN_PERSON_W = 20          # минимальная ширина бокса для роли

# ==============================
#   ПАРАМЕТРЫ ДВИЖЕНИЯ/АКТИВНОСТИ
# ==============================

FPS = 25  # обновится из видео

STILL_MAX_SPEED = 10       # px/сек — стоит/почти не двигается
WALK_SPEED_MIN = 10        # px/сек — начинается ходьба
WALK_SPEED_MAX = 120       # px/сек — быстрые перемещения (на будущее, если пригодится)

MOTION_WINDOW = 5          # окно для скорости
ACTIVITY_WINDOW = 15       # окно для сглаживания активности

# ==============================
#        ЗОНЫ В КАДРЕ
# ==============================

# Прямоугольные зоны — можно оставить как есть, используются только для "walk"/"work" меток
WORK_ZONE = (0, 0, 1000, 1000)
WALK_ZONE = (0, 0, 1000, 1000)

# ==============================
#    ПОЛИГОН ЗОНЫ ПОЕЗДА
# ==============================

# НОРМАЛИЗОВАННЫЕ координаты полигона поезда (x,y в диапазоне [0..1] от ширины/высоты кадра)
TRAIN_POLYGON_NORM = [
    (0.48, 0.22),
    (0.59, 1),
    (1, 1),
    (0.53, 0.22),
]
TRAIN_POLYGON = []  # сюда положим пиксельные координаты после чтения видео

# ==============================
#       ТЕПЛОВАЯ КАРТА
# ==============================

HEATMAP_DOWNSCALE = 4  # 4 → карта ~ в 16 раз меньше по пикселям

# ==============================
#       ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def point_in_rect(x, y, rect):
    x_min, y_min, x_max, y_max = rect
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def get_zone_label(cx, cy):
    """
    Возвращает:
      'work' — рабочая зона
      'walk' — проход
      'other' — остальное
    """
    if point_in_rect(cx, cy, WORK_ZONE):
        return "work"
    if point_in_rect(cx, cy, WALK_ZONE):
        return "walk"
    return "other"


def point_in_polygon(x, y, polygon):
    """
    Классический ray casting алгоритм.
    x, y — точка
    polygon — список (x_i, y_i)
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Проверка, пересекает ли ребро луч справа от точки
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i

    return inside


def classify_person_crop(crop_bgr, cls_model, cls_names):
    """
    Классификация кропа человека через YOLO-cls.
    Возвращает:
      role_label – строковая метка (напр. 'worker', 'other')
      conf       – вероятность top1
    """
    result = cls_model(crop_bgr, imgsz=224, verbose=False)[0]
    probs = result.probs
    top1_id = int(probs.top1)
    top1_conf = float(probs.top1conf)
    role_label = cls_names[top1_id]
    return role_label, top1_conf


def update_track_role(state, frame_idx, new_role, new_conf):
    """
    Обновление роли:
      - игнорируем слабые предсказания
      - current_role = мода по последним ROLE_WINDOW меткам
    """
    role_state = state.setdefault("role", {
        "current_role": "",
        "current_conf": 0.0,
        "last_frame": -ROLE_RECLASSIFY_EVERY,
        "labels_history": [],
        "confs_history": [],
    })

    if new_conf < ROLE_CONF_MIN:
        role_state["last_frame"] = frame_idx
        return role_state

    labels = role_state["labels_history"]
    confs = role_state["confs_history"]

    labels.append(new_role)
    confs.append(new_conf)

    if len(labels) > ROLE_WINDOW:
        labels[:] = labels[-ROLE_WINDOW:]
        confs[:] = confs[-ROLE_WINDOW:]

    role_state["labels_history"] = labels
    role_state["confs_history"] = confs

    counts = Counter(labels)
    most_common_role, _ = counts.most_common(1)[0]

    role_confs = [c for r, c in zip(labels, confs) if r == most_common_role]
    avg_conf = sum(role_confs) / len(role_confs) if role_confs else new_conf

    role_state["current_role"] = most_common_role
    role_state["current_conf"] = avg_conf
    role_state["last_frame"] = frame_idx

    return role_state


def update_track_motion(state, frame_idx, cx, cy, h, fps):
    """
    Обновляет:
      - историю позиций
      - скорость (px/сек)
      (высоту можно оставить на будущее, сейчас не используем)
    """
    motion = state.setdefault("motion", {
        "positions": [],
        "speed": 0.0,
    })

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
    """
    Простая и понятная логика активности:
      - если человек в полигоне поезда → working
      - иначе:
          speed > WALK_SPEED_MIN  → walking
          speed < STILL_MAX_SPEED → idle
          иначе → walking (чтоб не плодить лишние состояния)
    """
    if in_train_zone:
        return "working"

    if speed > WALK_SPEED_MIN:
        return "walking"

    if speed < STILL_MAX_SPEED:
        return "idle"

    return "walking"


# ==============================
#               MAIN
# ==============================

def main():
    global WORK_ZONE, WALK_ZONE, FPS, TRAIN_POLYGON

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

    print(f"Видео: {width}x{height}, FPS={fps:.2f}")

    # Зоны "проход" и "работа" — для меток, не для логики working
    WALK_ZONE = (0, 0, int(width * 0.48), height)
    WORK_ZONE = (int(width * 0.48), 0, width, height)

    # Полигон поезда в пикселях
    TRAIN_POLYGON = [
        (int(x * width), int(y * height))
        for (x, y) in TRAIN_POLYGON_NORM
    ]

    print(f"WORK_ZONE: {WORK_ZONE}")
    print(f"WALK_ZONE: {WALK_ZONE}")
    print(f"TRAIN_POLYGON: {TRAIN_POLYGON}")

    # тепловая карта (downscaled)
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
    csv_writer.writerow([
        "frame", "track_id", "det_class_id", "det_class_name",
        "x1", "y1", "x2", "y2",
        "role", "role_conf",
        "speed_px_per_sec", "zone",
        "in_train_polygon", "activity"
    ])

    track_states = {}   # track_id -> state

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

    for frame_idx, result in enumerate(results_gen):
        frame = result.orig_img
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            out_writer.write(frame)
            continue

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

                # движение
                update_track_motion(state, frame_idx, cx, cy, h, fps)
                motion_state = state.get("motion", {})
                speed = motion_state.get("speed", 0.0)

                # роль
                role_state = state.setdefault("role", {
                    "current_role": "",
                    "current_conf": 0.0,
                    "last_frame": -ROLE_RECLASSIFY_EVERY,
                    "labels_history": [],
                    "confs_history": [],
                })

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
                        role_label, conf = classify_person_crop(crop, cls_model, cls_names)
                        role_state = update_track_role(state, frame_idx, role_label, conf)

                current_role = role_state["current_role"]
                role_conf = role_state["current_conf"]

                zone = get_zone_label(cx, cy)
                in_train_poly = point_in_polygon(cx, cy, TRAIN_POLYGON)

                # активность + сглаживание по окну
                activity_state = state.setdefault("activity", {
                    "current_activity": "",
                    "history": [],
                })
                raw_activity = infer_activity(zone, speed, in_train_poly)

                hist = activity_state["history"]
                hist.append(raw_activity)
                if len(hist) > ACTIVITY_WINDOW:
                    hist[:] = hist[-ACTIVITY_WINDOW:]

                counts = Counter(hist)
                current_activity, _ = counts.most_common(1)[0]
                activity_state["current_activity"] = current_activity

                # тепловая карта: все, кто "working" (без проверки роли)
                if current_activity == "working":
                    hx = int(cx / HEATMAP_DOWNSCALE)
                    hy = int(cy / HEATMAP_DOWNSCALE)
                    if 0 <= hx < heatmap_w and 0 <= hy < heatmap_h:
                        heatmap[hy, hx] += 1.0

            # --- цвет бокса ---
            if det_class_name == "person":
                if current_activity == "working":
                    color = (0, 255, 0)        # зелёный — работает
                elif current_activity == "walking":
                    color = (255, 255, 0)      # жёлтый — идёт
                elif current_activity == "idle":
                    color = (0, 215, 255)      # оранжевый — стоит
                else:
                    color = (0, 255, 255)      # если что-то не определилось
            elif det_class_name == "train":
                color = (0, 0, 255)            # красный — поезд
            else:
                color = (255, 0, 255)

            # --- рисуем ---
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
                frame, label_text, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
            )

            # --- лог ---
            csv_writer.writerow([
                frame_idx,
                int(track_id) if track_id != -1 else -1,
                class_id,
                det_class_name,
                x1, y1, x2, y2,
                current_role,
                f"{role_conf:.3f}",
                f"{speed:.1f}",
                zone,
                int(in_train_poly),
                current_activity,
            ])

        out_writer.write(frame)

    print("Обработка видео завершена.")

    out_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # ==============================
    #   ГЕНЕРАЦИЯ ТЕПЛОВОЙ КАРТЫ
    # ==============================

    if np.max(heatmap) > 0:
        heat_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heat_uint8 = heat_norm.astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color_resized = cv2.resize(heat_color, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(OUTPUT_HEATMAP_PATH, heat_color_resized)
        print(f"Тепловая карта сохранена в: {OUTPUT_HEATMAP_PATH}")
    else:
        print("Тепловая карта пустая (никто не был в состоянии 'working').")

    print(f"Видео сохранено в: {OUTPUT_VIDEO_PATH}")
    print(f"CSV лог:          {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
