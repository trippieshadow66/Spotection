import cv2, os, time, argparse
import urllib.request
import numpy as np

from src.db import get_lot_by_id

# Per-lot flip rules (keep as-is for now)
NEEDS_FLIP = {
    1: True,   # Lot 1 camera upside-down
    2: False,  # Lot 2 is upright
}

INTERVAL = 2.0  # seconds between images


def fetch_jpeg(url):
    """JPEG snapshot grab (for cameras like lot 2)."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        img_array = np.asarray(bytearray(data), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("Snapshot fetch error:", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Spotection frame capture")
    parser.add_argument("--lot", type=int, default=1)
    args = parser.parse_args()

    lot_id = args.lot
    lot_info = get_lot_by_id(lot_id)

    if not lot_info:
        raise RuntimeError(f"No lot with id={lot_id} found in database. Did you create it via the admin site?")

    url = lot_info["stream_url"]
    if not url:
        raise RuntimeError(f"Lot {lot_id} has no stream_url set.")

    is_jpeg_camera = url.lower().endswith(".jpg") or "cgi-bin" in url.lower()

    base_dir = os.path.join("data", f"lot{lot_id}")
    output_dir = os.path.join(base_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Connecting to live stream for lot {lot_id}: {url}")

    cap = None
    if not is_jpeg_camera:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Unable to open MJPEG stream.")

    print(f"Capturing frames for lot {lot_id} every {INTERVAL}s\n")

    try:
        while True:
            if is_jpeg_camera:
                frame = fetch_jpeg(url)
                if frame is None:
                    print("Snapshot failed, retrying...")
                    time.sleep(1)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("MJPEG read failed, reconnecting...")
                    time.sleep(1)
                    cap = cv2.VideoCapture(url)
                    continue

            # Apply flip rule
            if NEEDS_FLIP.get(lot_id, False):
                frame = cv2.flip(frame, 0)

            ts = int(time.time() * 1000)
            save_path = os.path.join(output_dir, f"live_{ts}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"[Lot {lot_id}] Saved {save_path}")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
