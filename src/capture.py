import cv2, os, time

LIVE_STREAM_URL = "https://taco-about-python.com/video_feed"
OUTPUT_DIR = "data/frames"
INTERVAL = 2.0  # seconds between snapshots

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Connecting to live stream: {LIVE_STREAM_URL}")
    cap = cv2.VideoCapture(LIVE_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError(" Unable to open live feed.")

    print(" Capturing frames every", INTERVAL, "seconds. Press Ctrl+C to stop.\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(" Frame read failed. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(LIVE_STREAM_URL)
                continue

            ts = int(time.time() * 1000)
            path = os.path.join(OUTPUT_DIR, f"live_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
