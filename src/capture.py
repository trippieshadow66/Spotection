import cv2, os, time, argparse
import urllib.request
import numpy as np

# Lot stream URLs
LOT_STREAMS = {
    1: "https://taco-about-python.com/video_feed",  # MJPEG stream
    2: "http://170.249.152.2:8080/cgi-bin/viewer/video.jpg",  # JPEG snapshot
}

# Per-lot flip rules
NEEDS_FLIP = {
    1: True,   # Lot 1 camera upside-down
    2: False,  # Lot 2 is upright
}

INTERVAL = 2.0  # seconds between images


# JPEG snapshot grab (for lot 2)
def fetch_jpeg(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        img_array = np.asarray(bytearray(data), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Spotection frame capture")
    parser.add_argument("--lot", type=int, default=1)
    args = parser.parse_args()

    lot_id = args.lot
    url = LOT_STREAMS.get(lot_id)
    is_jpeg_camera = url.endswith(".jpg") or "cgi-bin" in url.lower()

    output_dir = os.path.join("data", f"lot{lot_id}", "frames")
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
