import sqlite3, json, os
from datetime import datetime

DB_PATH = os.path.join("data", "spotection.db")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lot_config (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        config_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS detection_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_path TEXT,
        overlay_path TEXT,
        timestamp TEXT,
        occupied_count INTEGER,
        free_count INTEGER,
        stall_status_json TEXT
    )""")
    conn.commit(); conn.close()
    print("Database initialized at", DB_PATH)

def save_lot_config(name, config_dict):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("INSERT INTO lot_config (name, config_json) VALUES (?, ?)",
                (name, json.dumps(config_dict)))
    conn.commit(); conn.close()

def get_latest_lot_config():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT config_json FROM lot_config ORDER BY id DESC LIMIT 1")
    row = cur.fetchone(); conn.close()
    return json.loads(row[0]) if row else None

def save_detection_result(frame_path, overlay_path, occupied_count, free_count, stall_status):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        INSERT INTO detection_results 
        (frame_path, overlay_path, timestamp, occupied_count, free_count, stall_status_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (frame_path, overlay_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          occupied_count, free_count, json.dumps(stall_status)))
    conn.commit(); conn.close()

def get_latest_detection():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT * FROM detection_results ORDER BY id DESC LIMIT 1")
    row = cur.fetchone(); conn.close()
    if not row: return None
    keys = ["id","frame_path","overlay_path","timestamp","occupied_count","free_count","stall_status_json"]
    rec = dict(zip(keys, row))
    rec["stall_status_json"] = json.loads(rec["stall_status_json"])
    return rec

if __name__ == "__main__":
    init_db()
