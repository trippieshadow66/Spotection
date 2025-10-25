from flask import Flask, render_template, jsonify, send_file
from src.db import get_latest_detection, init_db
import os, time

app = Flask(__name__)

# --- Folder paths ---
OVERLAY_DIR = "overlays"
MAP_DIR = "maps"
FALLBACK_IMAGE = "static/img/fallback.jpg"

# --- Initialize database ---
init_db()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_latest_overlay_path():
    """Return most recent overlay image."""
    try:
        if not os.path.exists(OVERLAY_DIR):
            os.makedirs(OVERLAY_DIR, exist_ok=True)
            return None
        jpgs = [f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith(".jpg")]
        if not jpgs:
            return None
        latest = max(jpgs, key=lambda f: os.path.getmtime(os.path.join(OVERLAY_DIR, f)))
        return os.path.join(OVERLAY_DIR, latest)
    except Exception as e:
        print(f"❌ Overlay lookup error: {e}")
        return None


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


# ------------------------------------------------------------
# Core Data Access
# ------------------------------------------------------------
def get_latest_parking_data():
    """Fetch the latest detection record from SQLite."""
    try:
        latest = get_latest_detection()
        if not latest:
            print("⚠️ No detection results yet.")
            return {
                "available": 0,
                "total": 0,
                "percentage": 0,
                "last_updated": "No data",
                "has_overlay": False,
                "has_map": False,
            }

        occupied = latest["occupied_count"]
        free = latest["free_count"]
        total = occupied + free
        overlay_path = latest["overlay_path"]

        return {
            "available": free,
            "total": total,
            "percentage": (free / total) * 100 if total else 0,
            "last_updated": latest["timestamp"],
            "has_overlay": os.path.exists(overlay_path) if overlay_path else False,
            "has_map": bool(get_latest_map_path())
        }

    except Exception as e:
        print(f"❌ Error loading parking data: {e}")
        return {
            "available": 0,
            "total": 0,
            "percentage": 0,
            "last_updated": "Error",
            "has_overlay": False,
            "has_map": False
        }


# ------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/api/parking-data')
def parking_data():
    return jsonify(get_latest_parking_data())


@app.route('/overlay-image')
def overlay_image():
    """Serve the most recent overlay or fallback image."""
    try:
        latest = get_latest_overlay_path()
        if latest and os.path.exists(latest):
            print(f"🖼️ Serving overlay: {latest}")
            return send_file(latest, mimetype="image/jpeg")
        return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")
    except Exception as e:
        print(f"❌ Error serving overlay: {e}")
        return send_file(FALLBACK_IMAGE, mimetype="image/jpeg")


@app.route('/map-image')
def map_image():
    """Serve the most recent top-down map schematic."""
    try:
        latest = get_latest_map_path()
        if latest and os.path.exists(latest):
            print(f"🗺️ Serving map: {latest}")
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
    os.makedirs("overlays", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    print("\n🚗 Starting Spotection Flask Server")
    print("🌍 Visit: http://localhost:5000")
    print("⏹️  Stop with CTRL+C\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
