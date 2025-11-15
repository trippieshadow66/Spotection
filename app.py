from flask import Flask, render_template, jsonify, send_file, Response
from src.db import get_latest_detection, save_detection_result, init_db
from src.detect import load_config, detect_frame
import cv2, os, time
from ultralytics import YOLO

app = Flask(__name__)

LIVE_STREAM_URL = "https://taco-about-python.com/video_feed"
MAP_DIR = "maps"
FALLBACK_IMAGE = "static/img/fallback.jpg"
MODEL_PATH = "yolov8s.pt"

# --- Initialize DB and model ---
init_db()
model = YOLO(MODEL_PATH)


# ------------------------------------------------------------
# MJPEG Stream Generators
# ------------------------------------------------------------
def generate_raw_feed():
    cap = cv2.VideoCapture(LIVE_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError("Unable to open stream")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 0)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_overlay_feed():
    cap = cv2.VideoCapture(LIVE_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError("Unable to open stream")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        try:
            overlay, occupied = detect_frame(frame, model)
            save_detection_result(
                frame_path="live-feed",
                overlay_path="",
                occupied_count=sum(occupied.values()),
                free_count=len(occupied) - sum(occupied.values()),
                stall_status=occupied
            )
            _, buffer = cv2.imencode('.jpg', overlay)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Detection error: {e}")
            time.sleep(1)


# ------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/raw-feed')
def raw_feed():
    return Response(generate_raw_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/overlay-latest')
def overlay_latest():
    try:
        OVERLAY_DIR = "overlays"
        jpgs = [f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith(".jpg")]
        if not jpgs:
            return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")
        latest = max(jpgs, key=lambda f: os.path.getmtime(os.path.join(OVERLAY_DIR, f)))
        return send_file(os.path.join(OVERLAY_DIR, latest), mimetype="image/jpeg")
    except Exception as e:
        print("overlay latest error:", e)
        return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")



@app.route('/api/parking-data')
def parking_data():
    latest = get_latest_detection()
    if not latest:
        return jsonify({"available": 0, "total": 0, "percentage": 0, "last_updated": "No data"})
    occupied = latest["occupied_count"]
    free = latest["free_count"]
    total = occupied + free
    return jsonify({
        "available": free,
        "total": total,
        "percentage": (free / total) * 100 if total else 0,
        "last_updated": latest["timestamp"]
    })


@app.route('/map-image')
def map_image():
    try:
        if not os.path.exists(MAP_DIR):
            os.makedirs(MAP_DIR, exist_ok=True)
        jpgs = [f for f in os.listdir(MAP_DIR) if f.lower().endswith(".jpg")]
        if not jpgs:
            return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")
        latest = max(jpgs, key=lambda f: os.path.getmtime(os.path.join(MAP_DIR, f)))
        return send_file(os.path.join(MAP_DIR, latest), mimetype="image/jpeg")
    except Exception:
        return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")


# ------------------------------------------------------------
# Run Server
# ------------------------------------------------------------
if __name__ == '__main__':
    print("\n🚗 Starting Spotection Flask Server")
    print("🌍 Visit: http://localhost:5000")
    print("📺 Live feed: http://localhost:5000/live-feed")
    print("📹 Raw feed:  http://localhost:5000/raw-feed\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
