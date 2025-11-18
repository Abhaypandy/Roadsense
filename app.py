import os
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, Response
from ultralytics import YOLO

# ------------------------------
#   CONFIG
# ------------------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO("best.pt")   # your YOLOv12 model
current_video_source = None


# ------------------------------
#   STREAM GENERATOR
# ------------------------------
def generate_frames():
    global current_video_source

    # choose source
    if current_video_source == "camera":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(current_video_source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        annotated = results.plot()

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        # yield frame as multipart stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# ------------------------------
#   ROUTES
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    img = cv2.imread(save_path)

    # inference
    results = model(img)[0]
    annotated = results.plot()

    _, buf = cv2.imencode('.jpg', annotated)
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_source

    if 'video' not in request.files:
        return "No video uploaded"

    file = request.files['video']
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    current_video_source = save_path
    return redirect(url_for('result_video', filename=file.filename))


@app.route('/result_video/<filename>')
def result_video(filename):
    return render_template('result_video.html', filename=filename)


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global current_video_source
    current_video_source = "camera"
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------
#   RUN APP
# ------------------------------
if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)
