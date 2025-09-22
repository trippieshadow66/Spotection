# Spotection
A low-cost computer vision system that uses a single camera and AI to detect open and occupied parking spaces in real time, and displays results through a simple web interface.

Spotection is a senior capstone project (CS 490) at Marshall University.  
It is an **AI-based parking spot detection system** that uses a single camera feed to identify open and occupied parking spaces in real time.  

The goal is to provide students, faculty, staff, and visitors with a graphical view of available parking spots through a **web interface**, without needing costly sensors or hardware.

---

# Features (MVP)
- Capture frames from a live camera or video feed  
- Detect cars using AI (YOLO/OpenCV)  
- Map detections onto predefined parking spots  
- Display a simple parking lot grid (green = open, red = filled) via a Flask web app  

---

# Long-Term Goals
- Store and manage multiple parking lot layouts in a database  
- Improve detection accuracy (>80%) with custom training  
- Handle challenging conditions (rain, snow, motorcycles, poor lighting, trash in spots, etc.)  
- Deploy to different lots and camera angles with minimal setup  

---

# Tech Stack
- **Python 3.11+**  
- **OpenCV** (video capture & frame processing)  
- **YOLO / PyTorch** (object detection)  
- **Flask** (web interface)  
- **GitHub** (version control & collaboration)  

---
# Getting Started

Clone the Repository
git clone https://github.com/trippieshadow66/Spotection.git

cd Spotection

Create & Activate Virtual Environment

Windows: 
python -m venv venv
venv\Scripts\Activate.ps1

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

Install Dependencies Within virtual enviornment ctrl+shift+p select venv enviornemnt
python -m pip install --upgrade pip
pip install -r requirements.txt

Verify Installation
Run the test script to confirm everything is installed correctly:
python test_env.py
✅ If successful, you’ll see version numbers for Flask, OpenCV, NumPy, Torch, and Ultralytics.

Run the Capture Script
python src/capture.py

A live preview window will open.

Snapshots are saved to data/frames/ every 2 seconds.

Press q or Ctrl+C to stop.

If the camera doesn’t open: edit src/capture.py and set
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
→ Try index 1 if 0 fails.
