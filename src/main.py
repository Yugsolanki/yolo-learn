from ultralytics import YOLO
import cv2

model = YOLO(model="yolo26x.pt")

stream_url = 0
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: could not open video stream")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream disconnected or ended")
        break
    
    results = model.predict(source=frame, conf=0.4, verbose=False, classes=[74])
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        
        # Downstream logic to read time
    
    annotated_frame = results[0].plot()
    cv2.imshow("CCTV Clock Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()