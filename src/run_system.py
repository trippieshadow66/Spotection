import os, time, shutil, glob
from datetime import datetime
from src.detect import detect_frame
from src.db import save_detection_result
from src.detect import load_config

# Configuration

FRAMES_DIR = "data/frames"
PROCESSED_DIR = "data/processed"
OVERLAYS_DIR = "overlays"
CONFIG_FILE = "data/lot_config.json"

CHECK_INTERVAL = 2         # seconds between checks for new frames
MAX_PROCESSED_FILES = 200  # how many processed frames to keep
MAX_AGE_HOURS = 2          # delete files older than this
KEEP_LATEST = True         # always keep most recent overlay/frame



# Utility Functions

def ensure_dirs():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(OVERLAYS_DIR, exist_ok=True)


def get_new_frames():
    """Return sorted list of frame paths in data/frames."""
    frames = sorted([
        os.path.join(FRAMES_DIR, f)
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ], key=os.path.getmtime)
    return frames


def cleanup_old_files(folder, max_files=200, max_age_hours=2, keep_latest=True):
    """Delete old files by age and count, keeping newest if requested."""
    if not os.path.exists(folder):
        return

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: os.path.getmtime(x)
    )

    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    # Age-based cleanup
    for f in files[:-1] if keep_latest else files:
        if os.path.getmtime(f) < cutoff:
            try:
                os.remove(f)
                print(f" Deleted old file (> {max_age_hours}h): {f}")
            except Exception as e:
                print(f" Failed to delete {f}: {e}")

    # Count-based cleanup
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: os.path.getmtime(x)
    )
    while len(files) > max_files:
        oldest = files.pop(0)
        if keep_latest and oldest == files[-1]:
            break
        try:
            os.remove(oldest)
            print(f" Deleted excess file (> {max_files} files): {oldest}")
        except Exception as e:
            print(f" Failed to delete {oldest}: {e}")



# Main Run Loop

def main():
    ensure_dirs()

    # Ensure config exists
    if not os.path.exists(CONFIG_FILE):
        print(" No lot_config.json found.")
        print(" Run these first:")
        print("   1. python -m src.capture  (take a few images)")
        print("   2. python -m src.stall_config_poly  (define stalls)")
        return

    print(" Spotection System Running... (CTRL+C to stop)\n")
    processed = set()

    try:
        while True:
            frames = get_new_frames()

            for fp in frames:
                if fp in processed:
                    continue

                print(f" New frame detected: {fp}")

                try:
                    overlay_path, occupied = detect_frame(fp, return_map=True)

                    # Save DB entry for each processed frame
                    if overlay_path and occupied:
                        save_detection_result(
                            frame_path=fp,
                            overlay_path=overlay_path,
                            occupied_count=sum(occupied.values()),
                            free_count=len(occupied) - sum(occupied.values()),
                            stall_status=occupied
                        )
                        print(f" DB updated for {os.path.basename(fp)}")

                    # Move to processed
                    dest = os.path.join(PROCESSED_DIR, os.path.basename(fp))
                    shutil.move(fp, dest)
                    processed.add(fp)

                except Exception as e:
                    print(f" Error processing {fp}: {e}")

            # --- Cleanup cycles ---
            cleanup_old_files(PROCESSED_DIR, MAX_PROCESSED_FILES, MAX_AGE_HOURS, KEEP_LATEST)
            cleanup_old_files(OVERLAYS_DIR, MAX_PROCESSED_FILES, MAX_AGE_HOURS, KEEP_LATEST)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n System stopped by user.")


if __name__ == "__main__":
    main()
