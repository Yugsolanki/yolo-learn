from ultralytics import YOLO
import os

if __name__ == '__main__':
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../datasets")

    YAML_PATH = os.path.join(BASE_DIR, "unified_dataset/data.yml")
    
    RUN_NAME = "run_1"

    model = YOLO("yolo26x.pt")

    results = model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16, # set it to -1 to use the maximum batch size for your GPU
        device=0,
        project="clock_detector",
        name=RUN_NAME, 
        patience=20 # stop if no improvements for 20 epochs
    )

    print(f"Training complete! Best weights saved in: clock_detector/{RUN_NAME}/weights/best.pt")