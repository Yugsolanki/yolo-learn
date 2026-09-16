from ultralytics import YOLO
import cv2
from PIL import Image
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data")

model = YOLO(model="yolo26n.pt")

img = Image.open(os.path.join(DATASET_PATH, "red-clock.jpg"))
results = model.predict(source=img, save=True, save_txt=True)