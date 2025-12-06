import sys
sys.path.append(r"E:\e_storage\Roadsense\yolov12")

from ultralytics import YOLO
import cv2

model = YOLO(r"E:\e_storage\Roadsense\Pothole-Computer-Vision-Project\best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    annotated = results.plot()

    cv2.imshow("YOLOv12 Inference", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
