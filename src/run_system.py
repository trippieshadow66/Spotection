import os, time, shutil
from datetime import datetime
from src.detect import detect_frame

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
FRAMES_DIR = "data/frames"
PROCESSED_DIR = "data/processed"
CONFIG_FILE = "data/lot_config.json"
CHECK_INTERVAL = 2
MAX_PROCESSED_FILES = 200
MAX_AGE_HOURS = 2


# ------------------------------------------------------------
def ensure_dirs():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_new_frames():
    frames = sorted([
        f for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return [os.path.join(FRAMES_DIR, f) for f in frames]


def cleanup_old_processed(max_files=MAX_PROCESSED_FILES, max_age_hours=MAX_AGE_HOURS):
    files = sorted(
        [os.path.join(PROCESSED_DIR, f)
         for f in os.listdir(PROCESSED_DIR)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: os.path.getmtime(x)
    )
    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    for f in files:
        if os.path.getmtime(f) < cutoff:
            os.remove(f)
            print(f" Deleted old file (>{max_age_hours}h): {f}")

    # Re-check and limit by count
    files = sorted(
        [os.path.join(PROCESSED_DIR, f)
         for f in os.listdir(PROCESSED_DIR)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: os.path.getmtime(x)
    )
    while len(files) > max_files:
        oldest = files.pop(0)
        if os.path.exists(oldest):
            os.remove(oldest)
            print(f" Deleted excess file (>{max_files} files): {oldest}")


# ------------------------------------------------------------
def main():
    ensure_dirs()

    # Ensure configuration exists
    if not os.path.exists(CONFIG_FILE):
        print(" No lot_config.json found.")
        print(" Run these first:")
        print("   1. python -m src.capture  (take a few images)")
        print("   2. python -m src.stall_config_poly  (define stalls)")
        return

    print(" Spotection detection system running (CTRL+C to stop)\n")
    processed = set()

    try:
        while True:
            frames = get_new_frames()

            for fp in frames:
                if fp in processed:
                    continue

                print(f"  New frame detected: {fp}")
                try:
                    overlay_path = detect_frame(fp)
                    if overlay_path:
                        print(f" Overlay saved: {overlay_path}")
                    else:
                        print(f" No overlay produced for {fp}")

                    dest = os.path.join(PROCESSED_DIR, os.path.basename(fp))
                    shutil.move(fp, dest)
                    processed.add(fp)

                except Exception as e:
                    print(f" Error processing {fp}: {e}")

            cleanup_old_processed()
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n System stopped by user.")


if __name__ == "__main__":
    main()
