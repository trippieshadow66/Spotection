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

