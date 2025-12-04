import os, json, time, cv2, numpy as np, argparse
from ultralytics import YOLO
from collections import defaultdict, deque
from src.db import save_detection_result

try:
    from shapely.geometry import Polygon
    SHAPELY_OK = True
except:
    SHAPELY_OK = False

MODEL_PATH = "yolov8s.pt"
CONF = 0.2
VEHICLE_CLASSES = {2, 3, 5, 7}
CHECK_INTERVAL = 2
FILTER_MIN_AREA = 800
HISTORY_LEN = 3
KEEP = 5   # keep last overlays/maps

# How strict we are about overlap between a stall and a vehicle box
STALL_OVERLAP_FRAC = 0.3   # fraction of stall area that must be covered
BOX_OVERLAP_FRAC = 0.3     # OR fraction of vehicle box area

# We only use the bottom part of the vehicle box (tires area) for overlap
BOX_VERTICAL_FRACTION_FROM_TOP = 0.4  # ignore top 30% of box

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
    if not os.path.exists(folder):
        return
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    full = [os.path.join(folder, f) for f in files]
    for f in full[:-KEEP]:
        try:
            os.remove(f)
        except:
            pass


def load_config(lot_id):
    config_path, *_ = get_paths(lot_id)
    if not os.path.exists(config_path):
        print(f"[Detect] Lot {lot_id}: no lot_config.json yet")
        return []

    with open(config_path) as f:
        cfg = json.load(f)

    stalls = []
    for s in cfg.get("stalls", []):
        pts = np.array(s["points"], np.int32)
        entry = {"id": str(s["id"]), "pts": pts, "lane": s["lane"]}
        if SHAPELY_OK:
            entry["poly"] = Polygon(pts)
        stalls.append(entry)
    return stalls


def draw_map(stalls, occ, lot_id):
    _, _, _, maps = get_paths(lot_id)
    os.makedirs(maps, exist_ok=True)

    # If no stalls, draw a simple placeholder
    if not stalls:
        canvas = np.zeros((300, 600, 3), dtype=np.uint8) + 40
        cv2.putText(
            canvas, "No stalls configured",
            (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 2
        )
        ts = int(time.time() * 1000)
        out_path = os.path.join(maps, f"map_{ts}.jpg")
        cv2.imwrite(out_path, canvas)
        cleanup(maps)
        return out_path

    # Group by lane
    lanes = {}
    for s in stalls:
        lanes.setdefault(s["lane"], []).append(s)

    ordered_lanes = [lanes[k] for k in sorted(lanes.keys())]
    for lane in ordered_lanes:
        lane.sort(key=lambda st: np.mean(st["pts"][:, 1]))

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
            cv2.putText(
                canvas, stall["id"], (x1 + 10, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

    ts = int(time.time() * 1000)
    out_path = os.path.join(maps, f"map_{ts}.jpg")
    cv2.imwrite(out_path, canvas)
    cleanup(maps)
    return out_path


def detect_frame(frame_path, model, lot_id):
    img = cv2.imread(frame_path)
    if img is None:
        print(f"[Detect] Skipping unreadable frame: {frame_path}")
        time.sleep(0.5)
        return None

    stalls = load_config(lot_id)

    # Run YOLO
    res = model.predict(img, conf=CONF, imgsz=1280, verbose=False)[0]

    # Build vehicle boxes list (optionally using Shapely)
    boxes = []
    for b in res.boxes:
        if int(b.cls) in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            w, h = (x2 - x1), (y2 - y1)
            if w * h < FILTER_MIN_AREA:
                continue

            # Only keep the bottom portion of the box (tire/contact area)
            new_y1 = y1 + BOX_VERTICAL_FRACTION_FROM_TOP * h
            y1 = new_y1

            box_entry = {
                "coords": (x1, y1, x2, y2),
                "area": (x2 - x1) * (y2 - y1),
            }
            if SHAPELY_OK:
                box_entry["poly"] = Polygon(
                    [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                )
            boxes.append(box_entry)

    # Initial occupancy (before smoothing)
    occ = {s["id"]: False for s in stalls}

    for s in stalls:
        sid = s["id"]

        if SHAPELY_OK and "poly" in s:
            stall_poly = s["poly"]
            stall_area = max(stall_poly.area, 1.0)

            for box in boxes:
                inter_area = stall_poly.intersection(box["poly"]).area
                if inter_area <= 0:
                    continue

                stall_frac = inter_area / stall_area
                box_frac = inter_area / max(box["area"], 1.0)

                if (stall_frac >= STALL_OVERLAP_FRAC or
                        box_frac >= BOX_OVERLAP_FRAC):
                    occ[sid] = True
                    break
        else:
            # Fallback: rectangle overlap with thresholds
            sx, sy, sw, sh = cv2.boundingRect(s["pts"])
            stall_area = max(sw * sh, 1.0)

            for box in boxes:
                x1, y1, x2, y2 = box["coords"]
                ix1, iy1 = max(x1, sx), max(y1, sy)
                ix2, iy2 = min(x2, sx + sw), min(y2, sy + sh)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = (ix2 - ix1) * (iy2 - iy1)
                stall_frac = inter_area / stall_area

                if stall_frac >= STALL_OVERLAP_FRAC:
                    occ[sid] = True
                    break

    # --- Temporal smoothing using stall_history ---
    for sid, val in occ.items():
        stall_history[sid].append(val)
        history = stall_history[sid]
        # majority vote over last HISTORY_LEN frames
        occ[sid] = sum(history) >= (len(history) / 2.0)

    # Draw overlay
    out = img.copy()
    for s in stalls:
        color = (0, 0, 255) if occ[s["id"]] else (0, 255, 0)
        cv2.polylines(out, [s["pts"]], True, color, 2)
        cx = int(np.mean(s["pts"][:, 0]))
        cy = int(np.mean(s["pts"][:, 1]))
        cv2.putText(out, s["id"], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, _, overlays, _ = get_paths(lot_id)
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

    model = YOLO(MODEL_PATH)
    print(f"[Detect] Running detection for lot {lot_id}")

    last_mtime = None
    latest_path = os.path.join(frames_dir, "latest.jpg")

    while True:
        if not os.path.exists(latest_path):
            time.sleep(1)
            continue

        try:
            mtime = os.path.getmtime(latest_path)
        except FileNotFoundError:
            time.sleep(1)
            continue

        # Only run when latest.jpg was updated
        if last_mtime is not None and mtime <= last_mtime:
            time.sleep(0.5)
            continue

        last_mtime = mtime

        try:
            result = detect_frame(latest_path, model, lot_id)
            if result is None:
                continue

            overlay_path, occ, map_path = result
            save_detection_result(
                frame_path=latest_path,
                overlay_path=overlay_path,
                occupied_count=sum(occ.values()),
                free_count=len(occ) - sum(occ.values()),
                stall_status=occ,
                lot_id=lot_id,
            )
            print(f"[Detect] Lot {lot_id}: processed frame {latest_path}")
        except Exception as e:
            print(f"[Detect] ERROR {lot_id}: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
