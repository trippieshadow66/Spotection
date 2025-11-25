from flask import Flask, render_template, jsonify, send_file, Response, request, abort
from src.db import (
    get_latest_detection_for_lot,
    get_all_lots,
    create_lot,
    delete_lot,
    get_lot_by_id,
    init_db,
    update_lot,
)
from src.process_manager import process_manager

import cv2, os, time, shutil, urllib.request, numpy as np, json

app = Flask(__name__)

FALLBACK_IMAGE = "static/img/fallback.jpg"


# ==============================================================
# Utility Functions
# ==============================================================

def fetch_jpeg(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except:
        return None


def generate_raw_feed(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        raise RuntimeError("Lot missing")

    url = lot["stream_url"]
    flip = lot["flip"]
    is_jpeg = url.lower().endswith(".jpg") or "cgi-bin" in url.lower()

    if is_jpeg:
        while True:
            frame = fetch_jpeg(url)
            if frame is None:
                time.sleep(1)
                continue
            if flip:
                frame = cv2.flip(frame, 0)

            _, buf = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() +
                b"\r\n"
            )
    else:
        cap = cv2.VideoCapture(url)
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(1)
                cap = cv2.VideoCapture(url)
                continue

            if flip:
                frame = cv2.flip(frame, 0)

            _, buf = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() +
                b"\r\n"
            )


def get_latest_jpg(folder):
    if not os.path.exists(folder):
        return None
    jpgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    if not jpgs:
        return None
    return max(jpgs, key=os.path.getmtime)


# ==============================================================
# PAGE ROUTES
# ==============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/admin/lot-config/<int:lot_id>")
def lot_config_html(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        abort(404)
    return render_template("lot_config.html", lot_id=lot_id, lot_name=lot["name"])


# ==============================================================
# RAW FEEDS
# ==============================================================

@app.route("/raw-feed/<int:lot_id>")
def feed(lot_id):
    return Response(generate_raw_feed(lot_id),
        mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/raw-photo/<int:lot_id>")
def raw_photo(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        abort(404)

    url = lot["stream_url"]
    flip = lot["flip"]

    frame = fetch_jpeg(url)
    if frame is None:
        cap = cv2.VideoCapture(url)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")

    if flip:
        frame = cv2.flip(frame, 0)

    base = f"data/lot{lot_id}"
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "_config_snapshot.jpg")
    cv2.imwrite(path, frame)
    return send_file(path, mimetype="image/jpeg")


# ==============================================================
# MAP + OVERLAY IMAGE
# ==============================================================

@app.route("/overlay-latest/<int:lot_id>")
def overlay_latest(lot_id):
    p = get_latest_jpg(f"data/lot{lot_id}/overlays")
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/map-image/<int:lot_id>")
def map_image(lot_id):
    p = get_latest_jpg(f"data/lot{lot_id}/maps")
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


# ==============================================================
# PARKING STATS
# ==============================================================

def _parking_data(lot_id):
    d = get_latest_detection_for_lot(lot_id)
    if not d:
        return {"available": 0, "total": 0, "percentage": 0, "last_updated": "No data"}

    occ = d["occupied_count"]
    free = d["free_count"]
    total = occ + free

    return {
        "available": free,
        "total": total,
        "percentage": (free / total) * 100 if total else 0,
        "last_updated": d["timestamp"],
    }


@app.route("/api/parking-data/<int:lot_id>")
def api_parking(lot_id):
    return jsonify(_parking_data(lot_id))


# ==============================================================
# LOT MANAGEMENT
# ==============================================================

@app.route("/api/lots", methods=["GET"])
def api_get_lots():
    lots = get_all_lots()
    out = []
    for lot in lots:
        lid = lot["id"]
        out.append({
            "id": lot["id"],
            "name": lot["name"],
            "totalSpots": lot["total_spots"],
            "flip": lot["flip"],
            "rawFeed": f"/raw-feed/{lid}",
            "overlayFeed": f"/overlay-latest/{lid}",
            "mapImage": f"/map-image/{lid}",
            "apiEndpoint": f"/api/parking-data/{lid}",
        })
    return jsonify({"lots": out})


@app.route("/api/lots/<int:lot_id>", methods=["GET"])
def api_get_lot(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        return jsonify({"error": "Lot not found"}), 404
    return jsonify(lot)


@app.route("/api/lots", methods=["POST"])
def api_create_lot():
    data = request.get_json(force=True)

    name = data.get("name")
    camera = data.get("cameraUrl")
    total = data.get("totalSpots", 0)

    if not name or not camera:
        return jsonify({"error": "Missing name or camera"}), 400

    try:
        total = int(total)
    except:
        total = 0

    lot_id = create_lot(name, camera, total)

    # Make folders
    base = f"data/lot{lot_id}"
    for sub in ("frames", "overlays", "maps"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # Make empty config
    cfg = os.path.join(base, "lot_config.json")
    if not os.path.exists(cfg):
        with open(cfg, "w") as f:
            f.write('{"stalls": []}')

    process_manager.start_lot(lot_id)

    return jsonify({"lot": get_lot_by_id(lot_id)}), 201


@app.route("/api/lots/<int:lot_id>", methods=["DELETE"])
def api_delete(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    process_manager.stop_lot(lot_id)
    delete_lot(lot_id)

    base = f"data/lot{lot_id}"
    shutil.rmtree(base, ignore_errors=True)

    return jsonify({"status": "ok"})


# ===== Flip API =====

@app.route("/api/lots/<int:lot_id>/flip", methods=["POST"])
def api_set_flip(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    data = request.get_json(force=True)
    flip = data.get("flip")

    if flip not in (0,1):
        return jsonify({"error": "flip must be 0 or 1"}), 400

    update_lot(lot_id, flip=flip)
    return jsonify({"status": "ok", "flip": flip})


# ===== Stall Config API =====

@app.route("/api/lots/<int:lot_id>/config", methods=["GET"])
def api_get_config(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    path = f"data/lot{lot_id}/lot_config.json"
    if not os.path.exists(path):
        return jsonify({"stalls": []})

    with open(path, "r") as f:
        return Response(f.read(), mimetype="application/json")


@app.route("/api/lots/<int:lot_id>/config", methods=["POST"])
def api_save_config(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    data = request.get_json(force=True)
    stalls = data.get("stalls")
    if stalls is None:
        return jsonify({"error": "Missing stalls"}), 400

    path = f"data/lot{lot_id}/lot_config.json"
    with open(path, "w") as f:
        json.dump({"stalls": stalls}, f, indent=2)

    return jsonify({"status": "ok"})


# ==============================================================
# Launch
# ==============================================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
