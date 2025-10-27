from flask import Flask, render_template, jsonify, send_file, Response
from src.db import get_latest_detection, save_detection_result, init_db
from src.detect import load_config, draw_overlay  # reuse your logic
import os, time, cv2, numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# --- Folder paths ---
MAP_DIR = "maps"
FALLBACK_IMAGE = "static/img/fallback.jpg"
LIVE_STREAM_URL = "https://taco-about-python.com/video_feed"
MODEL_PATH = "yolov8s.pt"
CONF = 0.15
VEHICLE_CLASSES = {2, 3, 5, 7}
FRAME_INTERVAL = 1.5  # seconds between detections

# --- Initialize database ---
init_db()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_latest_map_path():
    """Return most recent schematic map image."""
    try:
        if not os.path.exists(MAP_DIR):
            os.makedirs(MAP_DIR, exist_ok=True)
            return None
        jpgs = [f for f in os.listdir(MAP_DIR) if f.lower().endswith(".jpg")]
        if not jpgs:
            return None
        latest = max(jpgs, key=lambda f: os.path.getmtime(os.path.join(MAP_DIR, f)))
        return os.path.join(MAP_DIR, latest)
    except Exception as e:
        print(f"❌ Map lookup error: {e}")
        return None


def get_latest_parking_data():
    """Fetch the latest detection record from SQLite."""
    try:
        latest = get_latest_detection()
        if not latest:
            return {
                "available": 0, "total": 0, "percentage": 0,
                "last_updated": "No data", "has_map": False
            }

        occupied = latest["occupied_count"]
        free = latest["free_count"]
        total = occupied + free

        return {
            "available": free,
            "total": total,
            "percentage": (free / total) * 100 if total else 0,
            "last_updated": latest["timestamp"],
            "has_map": bool(get_latest_map_path())
        }

    except Exception as e:
        print(f"❌ Error loading parking data: {e}")
        return {
            "available": 0, "total": 0, "percentage": 0,
            "last_updated": "Error", "has_map": False
        }


# ------------------------------------------------------------
# 🔴 Live Feed with Overlays
# ------------------------------------------------------------
def generate_live_overlay_frames():
    """Stream live video feed with YOLO overlay annotations."""
    print(f"📡 Connecting to {LIVE_STREAM_URL}")
    cap = cv2.VideoCapture(LIVE_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open live stream: {LIVE_STREAM_URL}")

    stalls = load_config()
    model = YOLO(MODEL_PATH)

    last_detect = 0
    occupied_cache = {s["id"]: False for s in stalls}

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Lost connection, retrying...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(LIVE_STREAM_URL)
            continue

        now = time.time()
        if now - last_detect > FRAME_INTERVAL:
            last_detect = now
            try:
                # Run YOLO detection
                res = model.predict(frame, conf=CONF, verbose=False)[0]
                boxes = []
                for b in res.boxes:
                    if int(b.cls.item()) in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                        boxes.append((x1, y1, x2, y2))

                # --- Determine occupancy ---
                occupied = {s["id"]: False for s in stalls}
                for s in stalls:
                    contour = np.array(s["pts"], np.int32)
                    pid = s["id"]
                    sx, sy, sw, sh = cv2.boundingRect(contour)

                    for box in boxes[:]:
                        x1, y1, x2, y2 = box
                        shrink = 15
                        x1 += shrink; y1 += shrink
                        x2 -= shrink; y2 -= shrink
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2) + 25  # lower center point
                        ix1, iy1 = max(x1, sx), max(y1, sy)
                        ix2, iy2 = min(x2, sx + sw), min(y2, sy + sh)
                        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
                        overlap = (inter_w * inter_h) / (sw * sh + 1e-6)
                        in_poly = cv2.pointPolygonTest(contour, (cx, cy), False) >= 0
                        if in_poly or overlap > 0.25:
                            occupied[pid] = True
                            boxes.remove(box)
                            break

                occupied_cache = occupied
                frame = draw_overlay(frame, stalls, occupied_cache)

                # Optional: save to DB every few frames
                save_detection_result(
                    frame_path="live-feed",
                    overlay_path="",
                    occupied_count=sum(occupied.values()),
                    free_count=len(occupied) - sum(occupied.values()),
                    stall_status=occupied
                )
            except Exception as e:
                print(f"❌ Detection error: {e}")

        else:
            # Draw the last known overlay between YOLO updates
            frame = draw_overlay(frame, stalls, occupied_cache)

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/live-feed')
def live_feed():
    """Stream the live camera feed with overlay annotations."""
    print("🖥️ Streaming live overlay feed...")
    return Response(generate_live_overlay_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------------------------------------
# Web + API Routes
# ------------------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/api/parking-data')
def parking_data():
    return jsonify(get_latest_parking_data())


@app.route('/map-image')
def map_image():
    try:
        latest = get_latest_map_path()
        if latest and os.path.exists(latest):
            return send_file(latest, mimetype="image/jpeg")
        return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")
    except Exception as e:
        print(f"❌ Error serving map: {e}")
        return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")


# ------------------------------------------------------------
# Run Server
# ------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs("static/img", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    print("\n🚗 Starting Spotection Flask Server")
    print("🌍 Visit: http://localhost:5000")
    print("📺 Live feed: http://localhost:5000/live-feed")
    print("⏹️ Stop with CTRL+C\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
