import cv2
from ultralytics import YOLO
import os

class VideoProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def process_video(self, input, output):
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            annotated = results[0].plot()

            out.write(annotated)
        
        cap.release()
        out.release()

        return output