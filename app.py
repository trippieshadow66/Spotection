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



def get_single_frame_universal(url, flip=0):
    """
    Attempts to grab a single frame from ANY type of camera endpoint:
    - HTTP .jpg endpoints
    - MJPEG streams
    - RTSP streams
    - IP webcams
    """

    try:
        # Case 1 — direct JPEG snapshot URL
        if url.endswith(".jpg") or "jpg" in url.lower():
            req = urllib.request.urlopen(url, timeout=8)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        else:
            # Case 2 — RTSP or MJPEG or unknown
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print("[Grab] Failed to open stream")
                return None

            ok, img = cap.read()
            cap.release()
            if not ok:
                print("[Grab] Failed to read frame")
                return None

        if img is None:
            return None

        # Apply flip if required
        if flip:
            img = cv2.rotate(img, cv2.ROTATE_180)

        return img

    except Exception as e:
        print(f"[Grab] ERROR: {e}")
        return None
    
# ============================================================
# ALWAYS USE SAVED FRAMES — REMOVE LIVE FEED DEPENDENCY
# ============================================================

@app.route("/frame-latest/<int:lot_id>")
def frame_latest(lot_id):
    """Serve most recent captured frame for stall config + dashboard."""
    p = f"data/lot{lot_id}/frames/latest.jpg"
    if os.path.exists(p):
        return send_file(p, mimetype="image/jpeg")
    return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")


# ============================================================
# HELPERS
# ============================================================

def get_latest_jpg(folder):
    if not os.path.exists(folder):
        return None
    jpgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    if not jpgs:
        return None
    return max(jpgs, key=os.path.getmtime)


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/admin/lot-config/<int:lot_id>")
def lot_config_html(lot_id):
    lot = get_lot_by_id(lot_id)
    if not lot:
        abort(404)
    return render_template("lot_config.html", lot_id=lot_id, lot_name=lot["name"])


@app.route("/overlay-latest/<int:lot_id>")
def overlay_latest(lot_id):
    p = get_latest_jpg(f"data/lot{lot_id}/overlays")
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route("/map-image/<int:lot_id>")
def map_image(lot_id):
    p = get_latest_jpg(f"data/lot{lot_id}/maps")
    return send_file(p or FALLBACK_IMAGE, mimetype="image/jpeg")


# ============================================================
# PARKING DATA API
# ============================================================

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


# ============================================================
# LOT MANAGEMENT
# ============================================================

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
            "overlayFeed": f"/overlay-latest/{lid}",
            "mapImage": f"/map-image/{lid}",
            "rawFrame": f"/frame-latest/{lid}",   # <— new!
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

    base = f"data/lot{lot_id}"
    for sub in ("frames", "overlays", "maps"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

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


# ============================================================
# FLIP API
# ============================================================

@app.route("/api/lots/<int:lot_id>/flip", methods=["POST"])
def api_set_flip(lot_id):
    if not get_lot_by_id(lot_id):
        return jsonify({"error": "Lot not found"}), 404

    data = request.get_json(force=True)
    flip = data.get("flip")

    if flip not in (0, 1):
        return jsonify({"error": "flip must be 0 or 1"}), 400

    update_lot(lot_id, flip=flip)
    return jsonify({"status": "ok", "flip": flip})


# ============================================================
# STALL CONFIG API
# ============================================================

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


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":
    init_db()
    lots = get_all_lots()
    lot_ids = [lot["id"] for lot in lots]
    print(f"[PM] Auto-starting lots: {lot_ids}")
    process_manager.start_all(lot_ids)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
