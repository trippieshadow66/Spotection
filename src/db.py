import sqlite3, json, os
from datetime import datetime

DB_PATH = os.path.join("data", "spotection.db")


def _connect():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Initialize DB tables:
      - lot_config (legacy)
      - detection_results
      - lots (canonical)
    Patch: flip column added, seeding only once.
    """

    db_existed = os.path.exists(DB_PATH)
    conn = _connect()
    cur = conn.cursor()

    # -----------------------------
    # Legacy TABLE: lot_config
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lot_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            config_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -----------------------------
    # TABLE: detection_results
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_path TEXT,
            overlay_path TEXT,
            timestamp TEXT,
            occupied_count INTEGER,
            free_count INTEGER,
            stall_status_json TEXT
        )
    """)

    # Add lot_id column
    try:
        cur.execute("ALTER TABLE detection_results ADD COLUMN lot_id INTEGER DEFAULT 1")
        print("DB: Added lot_id column to detection_results")
    except sqlite3.OperationalError:
        pass

    # -----------------------------
    # TABLE: lots
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            stream_url TEXT NOT NULL,
            total_spots INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add flip column
    try:
        cur.execute("ALTER TABLE lots ADD COLUMN flip INTEGER DEFAULT 0")
        print("DB: Added flip column to lots")
    except sqlite3.OperationalError:
        pass

    conn.commit()

    # -----------------------------
    # Seed default lots ONLY IF DB is new
    # -----------------------------
    if not db_existed:
        print("DB: First-time initialization, seeding default lots")
        defaults = [
            ("West Campus Lot", "https://taco-about-python.com/video_feed", 45),
            ("East Campus Garage", "http://170.249.152.2:8080/cgi-bin/viewer/video.jpg", 60),
        ]
        for name, url, total in defaults:
            cur.execute(
                "INSERT INTO lots (name, stream_url, total_spots) VALUES (?, ?, ?)",
                (name, url, total)
            )
        conn.commit()

    conn.close()
    print("DB initialized.")


# ==============================================================
# Legacy lot_config helpers
# ==============================================================

def save_lot_config(name, config_dict):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO lot_config (name, config_json) VALUES (?, ?)",
        (name, json.dumps(config_dict))
    )
    conn.commit()
    conn.close()


def get_latest_lot_config():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT config_json FROM lot_config ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


# ==============================================================
# Detection results helpers
# ==============================================================

def save_detection_result(frame_path, overlay_path, occupied_count,
                          free_count, stall_status, lot_id=1):

    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO detection_results
        (frame_path, overlay_path, timestamp,
         occupied_count, free_count, stall_status_json, lot_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        frame_path, overlay_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        occupied_count, free_count,
        json.dumps(stall_status),
        lot_id
    ))
    conn.commit()
    conn.close()


def _row_to_dict(row):
    if not row:
        return None
    keys = [
        "id", "frame_path", "overlay_path", "timestamp",
        "occupied_count", "free_count", "stall_status_json", "lot_id"
    ]
    d = dict(zip(keys, row))
    d["stall_status_json"] = json.loads(d["stall_status_json"])
    return d


def get_latest_detection():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM detection_results ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


def get_latest_detection_for_lot(lot_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM detection_results WHERE lot_id = ? ORDER BY id DESC LIMIT 1",
        (lot_id,)
    )
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


# ==============================================================
# Lot CRUD helpers
# ==============================================================

def _lot_row_to_dict(row):
    if not row:
        return None
    keys = ["id", "name", "stream_url", "total_spots", "created_at", "flip"]
    return dict(zip(keys, row))


def get_all_lots():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, name, stream_url, total_spots, created_at, flip FROM lots ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return [_lot_row_to_dict(r) for r in rows]


def get_lot_by_id(lot_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, stream_url, total_spots, created_at, flip FROM lots WHERE id = ?",
        (lot_id,)
    )
    row = cur.fetchone()
    conn.close()
    return _lot_row_to_dict(row)


def create_lot(name, stream_url, total_spots=0) -> int:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO lots (name, stream_url, total_spots) VALUES (?, ?, ?)",
        (name, stream_url, total_spots)
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def update_lot(lot_id, name=None, stream_url=None, total_spots=None, flip=None):
    conn = _connect()
    cur = conn.cursor()

    fields = []
    params = []

    if name is not None:
        fields.append("name = ?")
        params.append(name)
    if stream_url is not None:
        fields.append("stream_url = ?")
        params.append(stream_url)
    if total_spots is not None:
        fields.append("total_spots = ?")
        params.append(total_spots)
    if flip is not None:
        fields.append("flip = ?")
        params.append(flip)

    if fields:
        params.append(lot_id)
        cur.execute(f"UPDATE lots SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()

    conn.close()


def delete_lot(lot_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM detection_results WHERE lot_id = ?", (lot_id,))
    cur.execute("DELETE FROM lots WHERE id = ?", (lot_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
