    Spotection
A low-cost computer vision system that uses a single camera and AI to detect open and occupied parking spaces in real time — displaying results through a live web dashboard.
Spotection is a senior capstone project (CS 490) at Marshall University.
It’s an AI-based parking spot detection system that uses YOLOv8 and OpenCV to identify open/occupied parking spaces, stores results in a local database, and visualizes them in a simple Flask web app.

 Features (MVP)
Capture frames from a webcam or video feed
Detect cars using YOLOv8 + OpenCV
Map detections onto pre-defined parking stalls
Store results in an SQLite database
Display both live overlays and a top-down schematic map
Real-time updates via Flask web interface

 Long-Term Goals
Support multiple parking lot configurations
Improve detection accuracy (>95%) with custom dataset training
Handle edge cases (lighting, rain, snow, motorcycles, occlusions)
Deploy to multiple cameras with minimal setup

 Tech Stack
Python 3.11+
OpenCV – video capture & image preprocessing
YOLOv8 / PyTorch – object detection
Flask – live web interface
SQLite3 – lightweight data storage
HTML/CSS/JS – dynamic web dashboard

    Setup Instructions
1 Clone the Repository
git clone https://github.com/trippieshadow66/Spotection.git
cd Spotection

2 Create & Activate a Virtual Environment (recommended can be done locally)

Windows (PowerShell):

python -m venv venv
venv\Scripts\Activate.ps1

macOS / Linux:

python3 -m venv venv
source venv/bin/activate
Then install dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt
Tip: In VS Code use Ctrl + Shift + P → “Python: Select Interpreter” → choose venv

3 Initialize the Database
python -m src.db
This creates data/spotection.db and initializes tables for:
Parking-lot configuration
Detection results

4 Capture starting Frames for setup
python -m src.capture
Opens a live camera preview.
Saves frames to data/frames/ every ~2 seconds.
Press q or Ctrl + C to stop.
If the camera doesn’t open, edit src/capture.py:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Try (1) or (2) if 0 fails

6 Configure Parking Stalls
python -m src.stall_config_poly
Start from the left most lane and the furthest stall coming towards the camera as you progress for map view.
Click around each stall (≥ 3 points).
(thin Rectangle down center of stall or both corners of front part of stall and back middle for triangle)
Press n to finish the stall.
Press a number key (1-9) to assign a lane (inside the window).
Repeat for each stall.
Press s to save → data/lot_config.json.
Press q to exit without saving


6 Start Continuous Detection
python run_system.py
This script will:
Watch data/frames/ for new images
Run YOLOv8 detection automatically
Save overlays → /overlays
Save schematic maps → /maps
Write results → data/spotection.db
Clean up older files (keeps last 2 hours / 200 files)
Run it while capture is active in another terminal:
# Terminal 1
python -m src.capture
# Terminal 2
python run_system.py

7 Launch the Web Dashboard
python app.py
Open your browser to 👉 http://localhost:5000
The dashboard will show:
Live detection overlay (from /overlays)
Top-down schematic map (from /maps)
Live stats: available / total / occupancy %
Auto-refresh every 3 seconds