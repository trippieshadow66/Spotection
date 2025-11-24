from flask import Flask, render_template, jsonify, send_file, Response, request, abort
from src.db import (
    get_latest_detection_for_lot,
    get_all_lots,
    create_lot,
    delete_lot,
    get_lot_by_id,
    init_db,
)
from src.process_manager import process_manager
import cv2, os, time, shutil, urllib.request, numpy as np, json

app = Flask(__name__)

FALLBACK_IMAGE = "static/img/fallback.jpg"

# Per-lot flip rules
NEEDS_FLIP = {
    1: True,
    2: False,
}

# ===================================================================
# Helpers
# ===================================================================

def fetch_jpeg(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print("fetch_jpeg error:", e)
        return None


def generate_raw_feed(lot_id: int):
    lot = get_lot_by_id(lot_id)
    if not lot:
        raise RuntimeError(f"Lot {lot_id} not found.")

    url = lot["stream_url"]
    is_jpeg = url.lower().endswith(".jpg") or "cgi-bin" in url.lower()

    if is_jpeg:
        while True:
            frame = fetch_jpeg(url)
            if frame is None:
                time.sleep(1)
                continue

            if NEEDS_FLIP.get(lot_id, False):
                frame = cv2.flip(frame, 0)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    else:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(1)
                continue

            if NEEDS_FLIP.get(lot_id, False):
                frame = cv2.flip(frame, 0)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )


def get_latest_jpg(folder):
    if not os.path.exists(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ===================================================================
# Page Routes
# ===================================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/admin/lot-config/<int:lot_id>")
def lot_config_page(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        abort(404)
    return render_template("lot_config.html", lot_id=lot_id, lot_name=lot["name"])


# ===================================================================
# Raw Feeds
# ===================================================================

@app.route("/raw-feed/<int:lot_id>")
def raw_feed_by_id(lot_id):
    return Response(
        generate_raw_feed(lot_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/raw-feed")
def raw_feed_1():
    return raw_feed_by_id(1)


@app.route("/raw-feed-2")
def raw_feed_2():
    return raw_feed_by_id(2)


@app.route("/raw-photo/<int:lot_id>")
def raw_photo(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        abort(404)

    url = lot["stream_url"]
    frame = fetch_jpeg(url)

    if frame is None:
        cap = cv2.VideoCapture(url)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")

    if NEEDS_FLIP.get(lot_id, False):
        frame = cv2.flip(frame, 0)

    base = os.path.join("data", f"lot{lot_id}")
    os.makedirs(base, exist_ok=True)
    temp_path = os.path.join(base, "_config_snapshot.jpg")
    cv2.imwrite(temp_path, frame)
    return send_file(temp_path, mimetype="image/jpeg")


# ===================================================================
# Overlay + Map
# ===================================================================

@app.route("/overlay-latest/<int:lot_id>")
def overlay_latest(lot_id):
    path = get_latest_jpg(f"data/lot{lot_id}/overlays")
    return send_file(path or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/map-image/<int:lot_id>")
def map_latest(lot_id):
    path = get_latest_jpg(f"data/lot{lot_id}/maps")
    return send_file(path or FALLBACK_IMAGE, mimetype="image/jpeg")


# Backward compatibility
@app.route("/overlay-latest")
def overlay1():
    return overlay_latest(1)

@app.route("/overlay-latest-2")
def overlay2():
    return overlay_latest(2)


@app.route("/map-image")
def map1():
    return map_latest(1)

@app.route("/map-image-2")
def map2():
    return map_latest(2)


# ===================================================================
# Parking Stats API
# ===================================================================

def _parking_data(lot_id: int):
    d = get_latest_detection_for_lot(lot_id)
    if not d:
        return {
            "available": 0,
            "total": 0,
            "percentage": 0,
            "last_updated": "No data",
        }

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
def api_stats(lot_id):
    if not get_lot_by_id(lot_id):
        abort(404)
    return jsonify(_parking_data(lot_id))


# ===================================================================
# Lot Management (Admin)
# ===================================================================

@app.route("/api/lots", methods=["GET"])
def api_get_lots():
    lots = get_all_lots()
    out = []
    for lot in lots:
        lid = lot["id"]
        out.append({
            "id": lid,
            "name": lot["name"],
            "totalSpots": lot["total_spots"] or 0,
            "rawFeed": f"/raw-feed/{lid}",
            "overlayFeed": f"/overlay-latest/{lid}",
            "mapImage": f"/map-image/{lid}",
            "apiEndpoint": f"/api/parking-data/{lid}",
        })
    return jsonify({"lots": out})


@app.route("/api/lots", methods=["POST"])
def api_create_lot():
    """
    Create a new lot from JSON body.

    Accepts either:
      { "name": "...", "cameraUrl": "...", "totalSpots": 50 }
    or:
      { "name": "...", "camera_url": "...", "total_spots": 50 }
    """
    data = request.get_json(force=True) or {}

    name = data.get("name")
    camera = data.get("cameraUrl") or data.get("camera_url")
    total = data.get("totalSpots")
    if total is None:
        total = data.get("total_spots", 0)

    if not name or not camera:
        return jsonify({"error": "Missing name or camera URL"}), 400

    try:
        total = int(total)
    except Exception:
        total = 0

    lot_id = create_lot(name, camera, total)

    # Make folders
    base = f"data/lot{lot_id}"
    for sub in ("frames", "overlays", "maps"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # Create empty stall config
    config_path = os.path.join(base, "lot_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write('{"stalls": []}\n')

    # Start processes for this lot 🎉
    process_manager.start_lot(lot_id)

    lot = get_lot_by_id(lot_id)

    payload = {
        "id": lot_id,
        "name": lot["name"],
        "totalSpots": lot["total_spots"] or 0,
        "rawFeed": f"/raw-feed/{lot_id}",
        "overlayFeed": f"/overlay-latest/{lot_id}",
        "mapImage": f"/map-image/{lot_id}",
        "apiEndpoint": f"/api/parking-data/{lot_id}",
    }

    # Keep old shape { "lot": {...} } for safety
    return jsonify({"lot": payload}), 201


@app.route("/api/lots/<int:lot_id>", methods=["DELETE"])
def api_delete_lot(lot_id):

    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    # Stop running scripts
    process_manager.stop_lot(lot_id)

    # Remove database rows
    delete_lot(lot_id)

    # Remove folders
    base = f"data/lot{lot_id}"
    if os.path.exists(base):
        shutil.rmtree(base, ignore_errors=True)

    return jsonify({"status": "ok"})


# ===================================================================
# Stall Config (GET + POST)
# ===================================================================

@app.route("/api/lots/<int:lot_id>/config", methods=["GET"])
def api_get_lot_config(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    path = f"data/lot{lot_id}/lot_config.json"
    if not os.path.exists(path):
        return jsonify({"stalls": []})

    with open(path, "r", encoding="utf-8") as f:
        try:
            return Response(f.read(), mimetype="application/json")
        except Exception:
            return jsonify({"stalls": []})


@app.route("/api/lots/<int:lot_id>/config", methods=["POST"])
def api_save_lot_config(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    data = request.get_json(force=True) or {}
    stalls = data.get("stalls")
    if stalls is None:
        return jsonify({"error": "Missing stalls"}), 400

    base = f"data/lot{lot_id}"
    os.makedirs(base, exist_ok=True)
    config_path = os.path.join(base, "lot_config.json")

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"stalls": stalls}, f, indent=2)
            f.write("\n")
    except Exception as e:
        print("Error writing stall config:", e)
        return jsonify({"error": "Failed to save config"}), 500

    return jsonify({"status": "ok"})


# ===================================================================
# Main entry
# ===================================================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
