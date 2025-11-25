import cv2, os, time, argparse
import urllib.request
import numpy as np

from src.db import get_lot_by_id

INTERVAL = 2.0  # seconds between frames


def fetch_jpeg(url):
    """Fetch a single JPEG snapshot."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print("Snapshot error:", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Spotection Frame Capture")
    parser.add_argument("--lot", type=int, required=True)
    args = parser.parse_args()

    lot_id = args.lot
    lot_info = get_lot_by_id(lot_id)

    if not lot_info:
        raise RuntimeError(f"Lot {lot_id} not found in DB")

    url = lot_info["stream_url"]
    is_jpeg = url.lower().endswith(".jpg") or "cgi-bin" in url.lower()

    # --- Ensure dirs ---
    base = f"data/lot{lot_id}"
    frames = os.path.join(base, "frames")
    overlays = os.path.join(base, "overlays")
    maps = os.path.join(base, "maps")
    for d in (base, frames, overlays, maps):
        os.makedirs(d, exist_ok=True)

    print(f"[Capture] Lot {lot_id} stream: {url}")

    cap = None
    if not is_jpeg:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Cannot open MJPEG feed.")

    print(f"[Capture] Running every {INTERVAL}s")

    try:
        while True:
            # 🔥 RELOAD FLIP SETTING DYNAMICALLY EVERY LOOP
            lot_info = get_lot_by_id(lot_id)
            flip = lot_info.get("flip", 0)

            # ----- grab frame -----
            if is_jpeg:
                frame = fetch_jpeg(url)
                if frame is None:
                    time.sleep(1)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(1)
                    cap = cv2.VideoCapture(url)
                    continue

            # Apply flip
            if flip:
                frame = cv2.flip(frame, 0)

            # Save
            ts = int(time.time()*1000)
            save_path = os.path.join(frames, f"live_{ts}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"[Capture] Lot {lot_id}: {save_path}")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("[Capture] Stopped")
    finally:
        if cap:
            cap.release()


if __name__ == "__main__":
    main()
