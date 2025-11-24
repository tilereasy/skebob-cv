import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("models/yolo11s")

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = model(image_path)
    for result in results:
        result.show()
        
    return image_rgb

image_path = 'data/input/example.png'
processed_image = process_image(image_path)

plt.imshow(processed_image)
plt.axis('off')
plt.show()