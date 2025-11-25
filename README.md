AI-Powered Multi-Lot Parking Detection System

Marshall University – CS 490 Senior Capstone

Spotection is a fully dynamic, multi-lot parking detection system that uses YOLOv8, OpenCV, and a Flask web dashboard to automatically detect occupied vs. available parking stalls in real time.

It supports:

 Up to 20 spots per parking lot (maximum 1 camera per lot support)

 Add/remove lots live — no restarts

 Browser-based stall configuration

 Live overlays + schematic maps per lot

 Auto-managed capture and detection processes

 Per-lot camera flip toggle 

 SQLite database with real-time results

 Tech Stack

Python 3.11+

YOLOv8 / Ultralytics

OpenCV

Flask

SQLite3

HTML / CSS / JavaScript

Custom Background Process Manager

 System Architecture

                ┌───────────────────────────┐
                │        Flask UI          │
                │  - Add/remove lots       │
                │  - Live dashboard        │
                │  - Stall configuration   │
                │  - Flip toggle           │
                └──────────────┬───────────┘
                               │
                               ▼
               ┌───────────────────────────┐
               │       SQLite DB           │
               │ lots / flip / stalls      │
               │ detection history          │
               └──────────────┬───────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │           Process Manager                   │
        │  - Starts capture.py & detect.py per lot    │
        │  - Stops processes when lot is deleted      │
        └──────────────────────┬──────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌───────────────────┐                     ┌──────────────────────┐
│    capture.py      │  → frames/lot# →   │      detect.py        │
│ - pulls images     │                     │ - YOLOv8 detection   │
│ - applies flip     │                     │ - stall overlap       │
│ - saves frames     │                     │ - overlay / map       │
└───────────────────┘                     └──────────────────────┘

 Installation & Setup

1. Clone Repository

git clone https://github.com/trippieshadow66/Spotection.git
cd Spotection

2. Create Virtual Environment

Windows (PowerShell):

python -m venv venv
venv\Scripts\Activate.ps1

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. First Run — Initialize DB & Start System

Simply run:

python app.py

This will:

Initialize the database

Start the Flask dashboard

Prepare the system for adding lots

No other scripts need to be run manually.

 Using the Web Dashboard

Navigate to:
 http://localhost:5000

You can now:

Add new lots

Configure stall polygons

Toggle flip per lot

View overlays & maps

View live stats

 Adding a New Parking Lot

Open the dashboard

Click Add Lot

Provide:

Name

Camera URL (JPEG or MJPEG)

Total Spots

Submit

The system automatically:

Creates folders for that lot

Starts capture + detection processes

Prepares a blank stall configuration file

 No restart required.

 Configuring Parking Stalls

Click Configure on the desired lot

A live snapshot appears

Draw polygon points around each stall

Assign lane number

Save configuration

Use the Flip ON/OFF button if the camera view is upside down

 Detection Engine

For each lot:

Reads the latest frame

Applies flip (if enabled)

Runs YOLOv8 detection

Computes stall overlaps

Applies smoothing (history-based)

Saves overlays & maps

Writes result to SQLite

Dashboard refreshes every 3 seconds.

 Deleting a Lot

When a lot is removed:

capture.py + detect.py processes are stopped

database entries are deleted

folder data/lot# is deleted

dashboard updates instantly

No restart needed.

 Folder Structure

Spotection/
│ app.py
│ requirements.txt
│ README.md
│
├── src/
│   ├── db.py
│   ├── capture.py
│   ├── detect.py
│   └── process_manager.py
│
├── data/
│   ├── spotection.db
│   └── lot1/
│       ├── frames/
│       ├── overlays/
│       ├── maps/
│       └── lot_config.json
│
├── templates/
│   ├── home.html
│   └── lot_config.html
│
└── static/
    └── img/fallback.jpg

 First-Time Testing Checklist

Run python app.py

Add a new lot

Confirm frames appear in data/lot#/frames/

Configure stalls

Confirm overlays update

Confirm schematic map updates

Add a second lot (optional)

Delete a lot and confirm:

Processes stop

Folder deleted

Dashboard updates instantly