import cv2, os, time, argparse, urllib.request, numpy as np
from src.db import get_lot_by_id
from app import get_single_frame_universal


INTERVAL = 2.0  # seconds between frames


def main():
    parser = argparse.ArgumentParser(description="Spotection Frame Capture")
    parser.add_argument("--lot", type=int, required=True)
    args = parser.parse_args()

    lot_id = args.lot
    lot_info = get_lot_by_id(lot_id)

    if not lot_info:
        raise RuntimeError(f"Lot {lot_id} not found in DB")

    url = lot_info["stream_url"]

    # Create folders
    base = f"data/lot{lot_id}"
    frames = os.path.join(base, "frames")
    overlays = os.path.join(base, "overlays")
    maps = os.path.join(base, "maps")
    for d in (base, frames, overlays, maps):
        os.makedirs(d, exist_ok=True)

    print(f"[Capture] Lot {lot_id} stream: {url}")
    print(f"[Capture] Saving latest.jpg every {INTERVAL}s")

    try:
        while True:
            lot_info = get_lot_by_id(lot_id)
            flip = lot_info.get("flip", 0)

            frame = get_single_frame_universal(url, flip=flip)

            if frame is None:
                print(f"[Capture] Lot {lot_id}: FAILED frame grab")
                time.sleep(1)
                continue

            save_path = os.path.join(frames, "latest.jpg")
            cv2.imwrite(save_path, frame)
            print(f"[Capture] Lot {lot_id}: saved latest.jpg")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("[Capture] Stopped")

    except Exception as e:
        print(f"[Capture] ERROR in lot {lot_id}: {e}")


if __name__ == "__main__":
    main()
