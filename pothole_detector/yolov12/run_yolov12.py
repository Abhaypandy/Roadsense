import sys
import argparse
import cv2
from pathlib import Path

# Add YOLOv12 repo to sys.path
sys.path.append(r"E:\e_storage\Roadsense\yolov12")

from ultralytics import YOLO

# Load model
MODEL_PATH = r"E:\e_storage\Roadsense\Pothole-Computer-Vision-Project\best.pt"
model = YOLO(MODEL_PATH)

# -----------------------------------------------------------
# IMAGE INFERENCE
# -----------------------------------------------------------
def run_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image:", image_path)
        return

    results = model(img)[0]
    annotated = results.plot()

    save_path = "output_image.jpg"
    cv2.imwrite(save_path, annotated)
    print(f"✅ Saved result to {save_path}")

    cv2.imshow("YOLOv12 Image Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------------------
# VIDEO INFERENCE
# -----------------------------------------------------------
def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video:", video_path)
        return

    out_path = "output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)[0]
        annotated = results.plot()
        out.write(annotated)

        cv2.imshow("YOLOv12 Video Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎬 Saved processed video to {out_path}")

# -----------------------------------------------------------
# WEBCAM INFERENCE
# -----------------------------------------------------------
def run_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()

        cv2.imshow("YOLOv12 Live Camera", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument("--video", type=str, help="Path to a video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam stream")

    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    elif args.video:
        run_video(args.video)
    else:
        run_webcam()
