import sqlite3, json, os
from datetime import datetime

DB_PATH = os.path.join("data", "spotection.db")


def _connect():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Initialize all DB tables:
      - lot_config (legacy)
      - detection_results
      - lots (canonical)
    Also ensures detection_results.lot_id exists.

    PATCHED:
      - Prevents reseeding default lots once DB exists.
      - Only seeds on FIRST-EVER initialization.
    """

    # PATCH: Detect if DB existed BEFORE running init
    db_exists = os.path.exists(DB_PATH)

    conn = _connect()
    cur = conn.cursor()

    # Legacy lot_config table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lot_config (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        config_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    # Detection results table
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
    conn.commit()

    # Add lot_id column if missing
    try:
        cur.execute("ALTER TABLE detection_results ADD COLUMN lot_id INTEGER DEFAULT 1")
        conn.commit()
        print("DB: Added lot_id column to detection_results")
    except sqlite3.OperationalError:
        pass  # already exists

    # Lots table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        stream_url TEXT NOT NULL,
        total_spots INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

    # PATCHED SEEDING:
    # Only seed if DB DID NOT EXIST before init()
    if not db_exists:
        print("DB: Performing FIRST-TIME seeding of default lots.")

        default_lots = [
            # Your classmate's working feed — KEEP IT
            ("West Campus Lot", "https://taco-about-python.com/video_feed", 45),

            # East Campus garage — KEEP IT
            ("East Campus Garage", "http://170.249.152.2:8080/cgi-bin/viewer/video.jpg", 60),
        ]

        for name, url, total in default_lots:
            cur.execute(
                "INSERT INTO lots (name, stream_url, total_spots) VALUES (?, ?, ?)",
                (name, url, total)
            )
        conn.commit()

    conn.close()
    print("Database initialized at", DB_PATH)


# ------------------ Legacy lot_config helpers ------------------ #

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


# ------------------ Detection result helpers ------------------ #

def save_detection_result(frame_path, overlay_path, occupied_count,
                          free_count, stall_status, lot_id: int = 1):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO detection_results
        (frame_path, overlay_path, timestamp,
         occupied_count, free_count, stall_status_json, lot_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        frame_path,
        overlay_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        occupied_count,
        free_count,
        json.dumps(stall_status),
        lot_id,
    ))
    conn.commit()
    conn.close()


def _row_to_dict(row):
    if not row:
        return None
    keys = [
        "id",
        "frame_path",
        "overlay_path",
        "timestamp",
        "occupied_count",
        "free_count",
        "stall_status_json",
        "lot_id",
    ]
    rec = dict(zip(keys, row))
    rec["stall_status_json"] = json.loads(rec["stall_status_json"])
    return rec


def get_latest_detection():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM detection_results ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


def get_latest_detection_for_lot(lot_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM detection_results WHERE lot_id = ? ORDER BY id DESC LIMIT 1",
        (lot_id,)
    )
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


# ------------------ Lots management ------------------ #

def _lot_row_to_dict(row):
    if not row:
        return None
    keys = ["id", "name", "stream_url", "total_spots", "created_at"]
    return dict(zip(keys, row))


def get_all_lots():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, name, stream_url, total_spots, created_at FROM lots ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return [_lot_row_to_dict(r) for r in rows]


def get_lot_by_id(lot_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, stream_url, total_spots, created_at FROM lots WHERE id = ?",
        (lot_id,)
    )
    row = cur.fetchone()
    conn.close()
    return _lot_row_to_dict(row)


def create_lot(name: str, stream_url: str, total_spots: int | None = None) -> int:
    if total_spots is None:
        total_spots = 0
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO lots (name, stream_url, total_spots) VALUES (?, ?, ?)",
        (name, stream_url, total_spots)
    )
    lot_id = cur.lastrowid
    conn.commit()
    conn.close()
    return lot_id


def update_lot(lot_id: int, name: str | None = None,
               stream_url: str | None = None,
               total_spots: int | None = None):
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
    if not fields:
        conn.close()
        return
    params.append(lot_id)
    sql = "UPDATE lots SET " + ", ".join(fields) + " WHERE id = ?"
    cur.execute(sql, params)
    conn.commit()
    conn.close()


def delete_lot(lot_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM detection_results WHERE lot_id = ?", (lot_id,))
    cur.execute("DELETE FROM lots WHERE id = ?", (lot_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
