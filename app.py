from flask import Flask, render_template, jsonify, send_file
import json
import time
import os

app = Flask(__name__)

# Hardcoded overlay path
OVERLAY_PATH = r"C:\Cybersecurity\NCL\Spotection\overlays\overlay_1761333396328.jpg"
CONFIG = "data/lot_config.json"

print("🚀 Starting Spotection Flask App...")

def get_latest_parking_data():
    """Get the latest parking data"""
    try:
        # Get stall data from your existing config
        total_spots = 0
        try:
            with open(CONFIG, "r") as f:
                config_data = json.load(f)
                total_spots = len(config_data.get("stalls", []))
            print(f"🅿️ Found {total_spots} parking spots in config")
        except Exception as e:
            total_spots = 20  # Default fallback
            print(f"⚠️ Using default spots: {e}")
        
        # Simulate data
        available_spots = max(0, total_spots - 8)  # Simulate 8 occupied spots
        
        return {
            "available": available_spots,
            "total": total_spots,
            "percentage": (available_spots / total_spots) * 100 if total_spots > 0 else 0,
            "last_updated": time.strftime("%H:%M:%S"),
            "has_overlay": os.path.exists(OVERLAY_PATH)
        }
    except Exception as e:
        print(f"❌ Error in get_latest_parking_data: {e}")
        return {
            "available": 0,
            "total": 0,
            "percentage": 0,
            "last_updated": "Error",
            "has_overlay": False,
            "error": str(e)
        }

@app.route('/')
def home():
    """Serve your website"""
    print("🌐 Home page requested")
    return render_template('home.html')

@app.route('/api/parking-data')
def parking_data():
    """API endpoint for live data"""
    return jsonify(get_latest_parking_data())

@app.route('/overlay-image')
def overlay_image():
    """Serve the hardcoded overlay image"""
    try:
        if os.path.exists(OVERLAY_PATH):
            print(f"🖼️ Serving overlay: {OVERLAY_PATH}")
            return send_file(OVERLAY_PATH, mimetype='image/jpeg')
        else:
            print("❌ Overlay file not found, using fallback")
            return send_file('static/parking-lot.jpg', mimetype='image/jpeg')
    except Exception as e:
        print(f"❌ Error serving overlay: {e}")
        return send_file('static/parking-lot.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    # Create the folders Flask needs
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    print("✅ Starting Flask server...")
    print("🌍 Server will be available at: http://localhost:5000")
    print("⏹️  Press CTRL+C to stop the server")
    
    # Check if overlay file exists
    if os.path.exists(OVERLAY_PATH):
        print(f"✅ Overlay file found: {OVERLAY_PATH}")
    else:
        print(f"❌ Overlay file NOT found: {OVERLAY_PATH}")
    
    app.run(debug=True, port=5000, host='0.0.0.0')