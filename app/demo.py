import cv2
from ultralytics import YOLO
from video_processor import VideoProcessor

model = YOLO("models/yolo11s")

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = model(image_rgb)
    
    return results

processor = VideoProcessor("models/yolo11s")
output = processor.process_video(
    input = "data/input/example.mp4",
    output = "data/output/result.mp4"
)

print(f"{output} is the result video")
