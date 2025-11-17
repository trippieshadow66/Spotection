from flask import Flask, render_template, jsonify, send_file, Response
from src.db import get_latest_detection_for_lot
from src.detect import detect_frame
import cv2, os, time
import urllib.request
import numpy as np

app = Flask(__name__)

LOT_STREAMS = {
    1: "https://taco-about-python.com/video_feed",
    2: "http://170.249.152.2:8080/cgi-bin/viewer/video.jpg",
}

# Per-lot flip rules
NEEDS_FLIP = {
    1: True,
    2: False,
}

FALLBACK_IMAGE = "static/img/fallback.jpg"


def fetch_jpeg(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except:
        return None


def generate_raw_feed(lot_id):
    url = LOT_STREAMS[lot_id]
    is_jpeg = url.endswith(".jpg") or "cgi-bin" in url

    if is_jpeg:
        while True:
            frame = fetch_jpeg(url)
            if frame is None:
                continue

            if NEEDS_FLIP.get(lot_id, False):
                frame = cv2.flip(frame, 0)

            _, buffer = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    else:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Cannot open MJPEG stream")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if NEEDS_FLIP.get(lot_id, False):
                frame = cv2.flip(frame, 0)

            _, buffer = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def get_latest_jpg(folder):
    if not os.path.exists(folder):
        return None
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/raw-feed")
def raw_feed_1():
    return Response(generate_raw_feed(1), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/raw-feed-2")
def raw_feed_2():
    return Response(generate_raw_feed(2), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/overlay-latest")
def overlay_latest_1():
    folder = f"data/lot1/overlays"
    p = get_latest_jpg(folder)
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/overlay-latest-2")
def overlay_latest_2():
    folder = f"data/lot2/overlays"
    p = get_latest_jpg(folder)
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/map-image")
def map_1():
    folder = f"data/lot1/maps"
    p = get_latest_jpg(folder)
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/map-image-2")
def map_2():
    folder = f"data/lot2/maps"
    p = get_latest_jpg(folder)
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/api/parking-data")
def api_1():
    d = get_latest_detection_for_lot(1)
    if not d:
        return jsonify({"available":0, "total":0, "percentage":0, "last_updated":"No data"})
    occ = d["occupied_count"]
    free = d["free_count"]
    total = occ + free
    return jsonify({
        "available": free,
        "total": total,
        "percentage": (free/total)*100 if total else 0,
        "last_updated": d["timestamp"]
    })


@app.route("/api/parking-data-2")
def api_2():
    d = get_latest_detection_for_lot(2)
    if not d:
        return jsonify({"available":0, "total":0, "percentage":0, "last_updated":"No data"})
    occ = d["occupied_count"]
    free = d["free_count"]
    total = occ + free
    return jsonify({
        "available": free,
        "total": total,
        "percentage": (free/total)*100 if total else 0,
        "last_updated": d["timestamp"]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
