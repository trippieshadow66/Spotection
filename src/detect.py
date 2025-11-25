import os, json, time, cv2, numpy as np, argparse
from ultralytics import YOLO
from collections import defaultdict, deque
from src.db import save_detection_result, get_lot_by_id

try:
    from shapely.geometry import Polygon, box as shapely_box
    SHAPELY_OK = True
except:
    SHAPELY_OK = False

MODEL_PATH = "yolov8s.pt"
CONF = 0.25
VEHICLE_CLASSES = {2, 3, 5, 7}
CHECK_INTERVAL = 2
FILTER_MIN_AREA = 1200
HISTORY_LEN = 3
KEEP = 5  # Keep last N overlays + maps

stall_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))


def get_paths(lot_id):
    base = f"data/lot{lot_id}"
    return (
        os.path.join(base, "lot_config.json"),
        os.path.join(base, "frames"),
        os.path.join(base, "overlays"),
        os.path.join(base, "maps"),
    )


def ensure_dirs(lot_id):
    _, frames, overlays, maps = get_paths(lot_id)
    for d in (frames, overlays, maps):
        os.makedirs(d, exist_ok=True)


def cleanup(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    full = [os.path.join(folder, f) for f in files]
    for old in full[:-KEEP]:
        try:
            os.remove(old)
        except:
            pass


def load_config(lot_id):
    config_path, *_ = get_paths(lot_id)
    with open(config_path) as f:
        cfg = json.load(f)

    stalls = []
    for s in cfg["stalls"]:
        pts = np.array(s["points"], np.int32)
        entry = {"id": str(s["id"]), "pts": pts, "lane": s["lane"]}
        if SHAPELY_OK:
            entry["poly"] = Polygon(pts)
        stalls.append(entry)
    return stalls


def draw_map(stalls, occ, lot_id):
    """Correct lane sorting and stall ordering (bottom to top)."""
    _, _, _, maps = get_paths(lot_id)
    os.makedirs(maps, exist_ok=True)

    # Group stalls by lane
    lanes = {}
    for s in stalls:
        lanes.setdefault(s["lane"], []).append(s)

    # Sort lanes: left → right (lane number)
    ordered_lanes = [lanes[k] for k in sorted(lanes.keys())]

    # Sort stalls bottom → top
    for lane in ordered_lanes:
        lane.sort(key=lambda st: np.mean(st["pts"][:, 1]))  # Y average

    # Draw map
    stall_w, stall_h = 130, 80
    pad_x, pad_y = 25, 25
    margin_x, margin_y = 50, 50

    rows = max(len(v) for v in ordered_lanes)
    cols = len(ordered_lanes)

    h = margin_y * 2 + rows * (stall_h + pad_y)
    w = margin_x * 2 + cols * (stall_w + pad_x)

    canvas = np.zeros((h, w, 3), dtype=np.uint8) + 40

    for col, lane in enumerate(ordered_lanes):
        for row, stall in enumerate(lane):
            x1 = margin_x + col * (stall_w + pad_x)
            y1 = margin_y + row * (stall_h + pad_y)
            x2, y2 = x1 + stall_w, y1 + stall_h

            color = (0, 0, 255) if occ[stall["id"]] else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.putText(canvas, stall["id"], (x1 + 10, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    ts = int(time.time() * 1000)
    out_path = os.path.join(maps, f"map_{ts}.jpg")
    cv2.imwrite(out_path, canvas)

    cleanup(maps)
    return out_path


def detect_frame(frame_path, model, lot_id):
    img = cv2.imread(frame_path)
    

    stalls = load_config(lot_id)
    res = model.predict(img, conf=CONF, imgsz=1280, verbose=False)[0]

    boxes = []
    for b in res.boxes:
        if int(b.cls) in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            if (x2 - x1) * (y2 - y1) >= FILTER_MIN_AREA:
                boxes.append((x1, y1, x2, y2))

    occ = {s["id"]: False for s in stalls}

    for s in stalls:
        sx, sy, sw, sh = cv2.boundingRect(s["pts"])
        for x1, y1, x2, y2 in boxes:
            ix1, iy1 = max(x1, sx), max(y1, sy)
            ix2, iy2 = min(x2, sx + sw), min(y2, sy + sh)
            if ix2 > ix1 and iy2 > iy1:
                occ[s["id"]] = True
                break

    # Draw overlay
    out = img.copy()
    for s in stalls:
        color = (0, 0, 255) if occ[s["id"]] else (0, 255, 0)
        cv2.polylines(out, [s["pts"]], True, color, 2)
        cx = int(np.mean(s["pts"][:, 0]))
        cy = int(np.mean(s["pts"][:, 1]))
        cv2.putText(out, s["id"], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, _, overlays, maps = get_paths(lot_id)
    ts = int(time.time() * 1000)
    overlay_path = os.path.join(overlays, f"overlay_{ts}.jpg")
    cv2.imwrite(overlay_path, out)

    cleanup(overlays)

    map_path = draw_map(stalls, occ, lot_id)

    return overlay_path, occ, map_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", type=int, required=True)
    args = parser.parse_args()

    lot_id = args.lot
    ensure_dirs(lot_id)

    _, frames_dir, _, _ = get_paths(lot_id)
    latest_path = os.path.join(frames_dir, "latest.jpg")

    model = YOLO(MODEL_PATH)
    print(f"[Detect] Running detection for lot {lot_id}")

    while True:
        if not os.path.exists(latest_path):
            time.sleep(1)
            continue

        try:
            overlay_path, occ, map_path = detect_frame(latest_path, model, lot_id)
            save_detection_result(
                frame_path=latest_path,
                overlay_path=overlay_path,
                occupied_count=sum(occ.values()),
                free_count=len(occ) - sum(occ.values()),
                stall_status=occ,
                lot_id=lot_id,
            )
        except Exception as e:
            print(f"[Detect] ERROR {lot_id}: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
