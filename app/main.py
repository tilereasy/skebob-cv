import cv2
import csv
import math
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# ==============================
#       НАСТРОЙКИ ПУТЕЙ
# ==============================

VIDEO_PATH = "data/input/video.mp4"                 # входное видео
OUTPUT_VIDEO_PATH = "data/output/depo_tracked.mp4"  # видео с боксами, ролями, активностями
OUTPUT_CSV_PATH = "output/depo_tracks.csv"     # лог в CSV

DET_MODEL_PATH = "models/yolo11m.pt"     # ДОобученная detect-модель (person/train)
CLS_MODEL_PATH = "models/yolo11s-cls"   # ДОобученная cls-модель ролей (worker/other/...)
TRACKER_CFG = "models/custom_bytetrack.yaml"            # конфиг трекера

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
WALK_SPEED_MAX = 120       # px/сек — быстрые перемещения

NEAR_TRAIN_MAX_DIST = 250  # px — ближе этого считаем "возле поезда"

MOTION_WINDOW = 5          # окно для вычисления скорости/дисперсии высоты
ACTIVITY_WINDOW = 15       # окно для сглаживания активности
ACTIVITY_RECALC_EVERY = 5  # как часто пересчитывать активность (в кадрах)

NEAR_TRAIN_MIN_FRAMES = 15    # минимум кадров подряд возле поезда для "working"
HEIGHT_VAR_WORK_MIN = 20.0    # порог дисперсии высоты для "working"

# ==============================
#        ЗОНЫ В КАДРЕ
# ==============================

# заглушки, реальные зоны зададим после чтения видео
WORK_ZONE = (0, 0, 1000, 1000)
WALK_ZONE = (0, 0, 1000, 1000)

# ==============================
#       ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def point_in_rect(x, y, rect):
    x_min, y_min, x_max, y_max = rect
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def get_zone_label(cx, cy):
    """
    Возвращает:
      'work' — рабочая зона возле поезда
      'walk' — проход
      'other' — остальное
    """
    if point_in_rect(cx, cy, WORK_ZONE):
        return "work"
    if point_in_rect(cx, cy, WALK_ZONE):
        return "walk"
    return "other"


def distance_to_train(cx, cy, train_bbox):
    """
    Расстояние от точки (cx, cy) до центра поезда.
    Если поезда нет — inf.
    """
    if train_bbox is None:
        return float("inf")
    x1t, y1t, x2t, y2t = train_bbox
    ctx = (x1t + x2t) / 2.0
    cty = (y1t + y2t) / 2.0
    return math.hypot(cx - ctx, cy - cty)


def classify_person_crop(crop_bgr, cls_model, cls_names):
    """
    Классификация кропа человека через твою YOLO-cls модель.
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
    Обновление роли в state["role"]:
      - игнорируем слабые предсказания
      - храним историю последних ROLE_WINDOW меток
      - current_role = мода по окну
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
        labels = labels[-ROLE_WINDOW:]
        confs = confs[-ROLE_WINDOW:]

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
      - историю высот bbox
      - скорость (px/сек)
      - дисперсию высот (height_var)
    """
    motion = state.setdefault("motion", {
        "positions": [],
        "heights": [],
        "speed": 0.0,
        "height_var": 0.0,
    })

    positions = motion["positions"]
    heights = motion["heights"]

    positions.append((frame_idx, cx, cy))
    heights.append(h)

    if len(positions) > MOTION_WINDOW:
        positions[:] = positions[-MOTION_WINDOW:]
        heights[:] = heights[-MOTION_WINDOW:]

    if len(positions) < 2:
        motion["speed"] = 0.0
        motion["height_var"] = 0.0
        return

    f0, x0, y0 = positions[0]
    f1, x1, y1 = positions[-1]

    df = max(1, f1 - f0)
    dist = math.hypot(x1 - x0, y1 - y0)
    speed_px_per_sec = dist * (fps / df)
    motion["speed"] = speed_px_per_sec

    if len(heights) >= 2:
        mean_h = sum(heights) / len(heights)
        var_h = sum((hh - mean_h) ** 2 for hh in heights) / len(heights)
        motion["height_var"] = var_h
    else:
        motion["height_var"] = 0.0


def infer_activity(is_worker, zone, speed, dist_train, height_var, near_train_frames):
    """
    Улучшенная логика активности:
      - учитывает длительность нахождения возле поезда (near_train_frames)
      - учитывает "ёрзание" по высоте bbox (height_var)
    """
    if not is_worker:
        return "irrelevant"

    is_near_train = (dist_train <= NEAR_TRAIN_MAX_DIST)

    # worker в рабочей зоне возле поезда
    if zone == "work" and is_near_train:
        # только что подошёл
        if near_train_frames < NEAR_TRAIN_MIN_FRAMES:
            if speed > WALK_SPEED_MIN:
                return "walking"
            else:
                return "idle"

        # достаточно долго у поезда
        if speed <= WALK_SPEED_MAX and height_var >= HEIGHT_VAR_WORK_MIN:
            return "working"

        if speed <= STILL_MAX_SPEED:
            return "idle"

        if speed > WALK_SPEED_MIN:
            return "walking"

        return "idle"

    # worker в проходе
    if zone == "walk":
        if speed > WALK_SPEED_MIN:
            return "walking"
        else:
            return "idle"

    # worker в других зонах
    if speed > WALK_SPEED_MIN:
        return "walking"
    else:
        return "idle"


def update_track_activity(state, frame_idx, is_worker, zone, speed, dist_train, height_var):
    """
    Обновляет state["activity"]:
      - поддерживает счётчик кадров возле поезда
      - считает "сырую" активность через infer_activity
      - сглаживает её по окну ACTIVITY_WINDOW
      - добавляет лёгкий гистерезис для 'working'
    """
    activity_state = state.setdefault("activity", {
        "current_activity": "idle",
        "history": [],
        "last_frame": -ACTIVITY_RECALC_EVERY,
        "near_train_frames": 0,
    })

    # обновляем near_train_frames каждый кадр
    if is_worker and zone == "work" and dist_train <= NEAR_TRAIN_MAX_DIST:
        activity_state["near_train_frames"] += 1
    else:
        activity_state["near_train_frames"] = 0

    if frame_idx - activity_state["last_frame"] < ACTIVITY_RECALC_EVERY:
        return activity_state

    near_train_frames = activity_state["near_train_frames"]

    raw_activity = infer_activity(
        is_worker, zone, speed, dist_train, height_var, near_train_frames
    )

    hist = activity_state["history"]
    hist.append(raw_activity)
    if len(hist) > ACTIVITY_WINDOW:
        hist[:] = hist[-ACTIVITY_WINDOW:]

    counts = Counter(hist)
    most_common_activity, _ = counts.most_common(1)[0]

    # гистерезис: не даём сразу "упасть" из working в idle/walking
    prev = activity_state["current_activity"]
    if prev == "working" and most_common_activity in ("idle", "walking"):
        num_non_work = sum(1 for a in hist if a != "working")
        if num_non_work < len(hist) // 2:
            most_common_activity = "working"

    activity_state["current_activity"] = most_common_activity
    activity_state["last_frame"] = frame_idx

    return activity_state


# ==============================
#               MAIN
# ==============================

def main():
    global WORK_ZONE, WALK_ZONE, FPS

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

    # Пример: рабочая зона — левая 70%, проход — правая 30%
    WORK_ZONE = (0, 0, int(width * 0.7), height)
    WALK_ZONE = (int(width * 0.7), 0, width, height)

    print(f"WORK_ZONE: {WORK_ZONE}")
    print(f"WALK_ZONE: {WALK_ZONE}")

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
        "speed_px_per_sec", "zone", "dist_to_train_px",
        "height_var", "activity"
    ])

    track_states = {}   # track_id -> state
    last_train_bbox = None

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

        # --- сначала найдём поезд в кадре ---
        train_bbox_this_frame = None
        for bbox, cls_id, track_id in zip(xyxy, cls_arr, ids):
            class_id = int(cls_id)
            if class_id == TRAIN_CLASS_ID:
                x1t, y1t, x2t, y2t = map(int, bbox)
                train_bbox_this_frame = (x1t, y1t, x2t, y2t)
                break

        if train_bbox_this_frame is not None:
            last_train_bbox = train_bbox_this_frame

        # --- обрабатываем все объекты ---
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
            height_var = 0.0
            zone = ""
            dist_train = float("inf")
            current_activity = ""

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
                height_var = motion_state.get("height_var", 0.0)

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
                dist_train = distance_to_train(cx, cy, last_train_bbox)
                is_worker = (current_role.lower() == "worker")

                activity_state = update_track_activity(
                    state, frame_idx, is_worker, zone, speed, dist_train, height_var
                )
                current_activity = activity_state["current_activity"]

            # --- цвет бокса ---
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
                f"{dist_train:.1f}" if dist_train != float("inf") else "",
                f"{height_var:.1f}",
                current_activity,
            ])

        out_writer.write(frame)

    print("Готово.")
    out_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Видео сохранено в: {OUTPUT_VIDEO_PATH}")
    print(f"CSV лог:          {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
