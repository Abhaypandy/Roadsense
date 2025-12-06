# ...existing code...
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    annotated = results.plot()

    cv2.imshow("YOLOv12", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# ...existing code...