from ultralytics import YOLO
import cv2
from PIL import Image
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data")

model = YOLO(model="yolo26n.pt")

results = model.predict(source="0", save=True, save_txt=True, conf=0.75, device="0", classes=[74])