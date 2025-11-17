import os, json, time, glob, cv2, numpy as np, argparse
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict, deque
from src.db import save_detection_result

try:
    from shapely.geometry import Polygon, box as shapely_box
    SHAPELY_OK = True
except:
    SHAPELY_OK = False

# Per-lot camera flip rules
NEEDS_FLIP = {
    1: True,
    2: False,
}

MODEL_PATH = "yolov8s.pt"
CONF = 0.25
VEHICLE_CLASSES = {2, 3, 5, 7}
CHECK_INTERVAL = 2
MAX_KEEP = 5
FILTER_MIN_AREA = 1200
HISTORY_LEN = 3
stall_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))


def get_paths(lot_id):
    base = os.path.join("data", f"lot{lot_id}")
    return (
        os.path.join(base, "lot_config.json"),
        os.path.join(base, "frames"),
        os.path.join(base, "overlays"),
        os.path.join(base, "maps"),
    )


def ensure_dirs(lot_id):
    _, frames, overlays, maps = get_paths(lot_id)
    for d in [frames, overlays, maps]:
        os.makedirs(d, exist_ok=True)


def load_config(lot_id):
    config_path, _, _, _ = get_paths(lot_id)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config for lot {lot_id}")

    with open(config_path) as f:
        cfg = json.load(f)

    stalls = []
    for s in cfg["stalls"]:
        pts = np.array(s["points"], dtype=np.int32)
        obj = {"id": str(s["id"]), "pts": pts, "lane": s["lane"]}
        if SHAPELY_OK:
            obj["poly"] = Polygon(pts)
        stalls.append(obj)
    return stalls


def cleanup(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=os.path.getmtime)
    for old in files[:-MAX_KEEP]:
        try: os.remove(old)
        except: pass


def compute_overlap(stall, box):
    x1,y1,x2,y2 = box
    if SHAPELY_OK:
        A = stall["poly"]
        B = shapely_box(x1,y1,x2,y2)
        inter = A.intersection(B).area
        return inter / max(A.area, 1)
    # bounding box fallback
    sx, sy, sw, sh = cv2.boundingRect(stall["pts"])
    ix1, iy1 = max(x1,sx), max(y1,sy)
    ix2, iy2 = min(x2,sx+sw), min(y2,sy+sh)
    inter_w, inter_h = max(0,ix2-ix1), max(0,iy2-iy1)
    return (inter_w*inter_h) / (sw*sh)


def assign_occupancy(stalls, boxes):
    occ = {s["id"]: False for s in stalls}
    for b in boxes:
        best = None
        best_ov = 0
        for s in stalls:
            ov = compute_overlap(s, b)
            if ov > best_ov:
                best_ov = ov
                best = s["id"]
        if best and best_ov >= 0.20:
            occ[best] = True
    return occ


def smooth(occ_raw):
    for pid, occ in occ_raw.items():
        stall_history[pid].append(1 if occ else 0)
    return {pid: sum(hist) >= len(hist)/2 for pid, hist in stall_history.items()}


def draw_overlay(img, stalls, occ, boxes):
    out = img.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0),2)
    for s in stalls:
        pts = s["pts"]
        color = (0,0,255) if occ[s["id"]] else (0,255,0)
        cv2.polylines(out, [pts], True, color, 2)
        cx, cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
        cv2.putText(out, s["id"], (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_map(stalls, occ, lot_id):
    _, _, _, maps = get_paths(lot_id)
    os.makedirs(maps, exist_ok=True)

    stall_w, stall_h = 120, 80
    pad_x, pad_y = 30, 25
    margin_x, margin_y = 80, 80

    lanes = {}
    for s in stalls:
        lanes.setdefault(s["lane"], []).append(s)
    for lane in lanes.values():
        lane.sort(key=lambda x: np.mean(x["pts"][:,1]))

    canvas_h = margin_y*2 + max(len(v) for v in lanes.values()) * (stall_h+pad_y)
    canvas_w = margin_x*2 + len(lanes) * (stall_w+pad_x)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 35

    lane_ids = sorted(lanes.keys())
    for col,lid in enumerate(lane_ids):
        for row, s in enumerate(lanes[lid]):
            x1 = margin_x + col*(stall_w+pad_x)
            y1 = margin_y + row*(stall_h+pad_y)
            x2, y2 = x1+stall_w, y1+stall_h
            color = (0,0,255) if occ[s["id"]] else (0,255,0)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),color,-1)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,255,255),2)

    ts = int(time.time()*1000)
    path = os.path.join(maps, f"map_{ts}.jpg")
    cv2.imwrite(path, canvas)
    cleanup(maps)
    return path


def detect_frame(input_image, model, lot_id):
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
    else:
        img = input_image.copy()

    # Apply flip if needed
    

    stalls = load_config(lot_id)
    result = model.predict(img, conf=CONF, imgsz=1280, verbose=False)[0]

    boxes = []
    for b in result.boxes:
        cls = int(b.cls.item())
        if cls in VEHICLE_CLASSES:
            x1,y1,x2,y2 = map(float, b.xyxy[0])
            if (x2-x1)*(y2-y1) >= FILTER_MIN_AREA:
                boxes.append((x1,y1,x2,y2))

    boxes = [b for b in boxes if any(compute_overlap(s, b)>0.05 for s in stalls)]
    occ = assign_occupancy(stalls, boxes)
    occ = smooth(occ)

    overlay = draw_overlay(img, stalls, occ, boxes)
    _, frames, overlays, maps = get_paths(lot_id)

    os.makedirs(overlays, exist_ok=True)
    ts = int(time.time()*1000)
    overlay_path = os.path.join(overlays, f"overlay_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay)

    draw_map(stalls, occ, lot_id)
    cleanup(overlays)

    return overlay_path, occ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", type=int, required=True)
    args = parser.parse_args()

    lot_id = args.lot
    ensure_dirs(lot_id)

    _, frames_dir, _, _ = get_paths(lot_id)

    if not os.path.exists(frames_dir):
        print("Missing frames folder")
        return

    model = YOLO(MODEL_PATH)

    print(f"Running detection for lot {lot_id}")

    while True:
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not frames:
            time.sleep(CHECK_INTERVAL)
            continue

        latest = frames[-1]
        print(f"[Lot {lot_id}] Processing {latest}")

        try:
            overlay_path, occ = detect_frame(latest, model, lot_id)
            save_detection_result(
                frame_path=latest,
                overlay_path=overlay_path,
                occupied_count=sum(occ.values()),
                free_count=len(occ)-sum(occ.values()),
                stall_status=occ,
                lot_id=lot_id,
            )
        except Exception as e:
            print(f"[Lot {lot_id}] Error:", e)

        for old in frames[:-1]:
            try: os.remove(old)
            except: pass

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
