import os, json, time, glob, shutil, cv2, numpy as np
from ultralytics import YOLO
from datetime import datetime
from src.db import save_detection_result

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CONFIG = "data/lot_config.json"
MODEL_PATH = "yolov8s.pt"
CONF = 0.15
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

FRAMES_DIR = "data/frames"
PROCESSED_DIR = "data/processed"
OVERLAYS_DIR = "overlays"
MAP_DIR = "maps"

CHECK_INTERVAL = 2
MAX_PROCESSED_FILES = 200
MAX_AGE_HOURS = 2
KEEP_LATEST = True


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def ensure_dirs():
    for d in [FRAMES_DIR, PROCESSED_DIR, OVERLAYS_DIR, MAP_DIR]:
        os.makedirs(d, exist_ok=True)


def load_config():
    """Load stall polygons (and lane numbers) from lot_config.json"""
    if not os.path.exists(CONFIG):
        raise FileNotFoundError("Missing lot_config.json. Run stall_config_poly.py first.")
    with open(CONFIG, "r") as f:
        cfg = json.load(f)
    stalls = []
    for s in cfg.get("stalls", []):
        pts = np.array(s["points"], dtype=np.int32)
        stalls.append({
            "id": str(s["id"]),
            "pts": pts,
            "lane": s.get("lane", 1)
        })
    return stalls


def draw_overlay(img, stalls, occupied_map, boxes=None):
    """Draw YOLO detections + stall overlays"""
    out = img.copy()

    # --- Draw YOLO boxes and center points ---
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(out, "C", (cx - 5, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- Stall outlines ---
    for s in stalls:
        pid, pts = s["id"], s["pts"]
        color = (0, 0, 255) if occupied_map.get(pid) else (0, 200, 0)
        cv2.polylines(out, [pts], True, color, 2)
        cX, cY = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        cv2.putText(out, pid, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Summary text ---
    tot, occ = len(stalls), sum(occupied_map.values())
    txt = f"Occupied: {occ}/{tot}   Free: {tot - occ}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 20, 10 + th + 10), (0, 0, 0), -1)
    cv2.putText(out, txt, (20, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def draw_topdown_map(stalls, occupied_map, output_dir=MAP_DIR):
    """Draw schematic preserving real lot layout (lanes left→right, stalls back→front)"""
    os.makedirs(output_dir, exist_ok=True)
    stall_w, stall_h = 120, 80
    pad_x, pad_y = 30, 25
    margin_x, margin_y = 80, 80
    bg_color = (35, 35, 35)

    lanes = {}
    for s in stalls:
        lanes.setdefault(s.get("lane", 1), []).append(s)

    lane_ids = sorted(lanes.keys())
    for lid in lane_ids:
        lanes[lid].sort(key=lambda s: np.mean(s["pts"][:, 1]))  # smaller Y = back

    max_rows = max(len(v) for v in lanes.values())
    cols = len(lane_ids)
    canvas = np.full(
        (int(margin_y * 2 + max_rows * (stall_h + pad_y)),
         int(margin_x * 2 + cols * (stall_w + pad_x)), 3), bg_color, np.uint8)

    for c, lid in enumerate(lane_ids):
        for r, s in enumerate(lanes[lid]):
            pid = s["id"]
            x1 = int(margin_x + c * (stall_w + pad_x))
            y1 = int(margin_y + r * (stall_h + pad_y))
            x2, y2 = x1 + stall_w, y1 + stall_h
            color = (0, 0, 255) if occupied_map.get(pid) else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(canvas, str(pid),
                        (x1 + stall_w // 2 - 10, y1 + stall_h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Lane {lid}",
                    (int(margin_x + c * (stall_w + pad_x) + 5), 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    ts = int(time.time() * 1000)
    map_path = os.path.join(output_dir, f"map_{ts}.jpg")
    cv2.imwrite(map_path, canvas)
    return map_path


# ------------------------------------------------------------
# YOLO Detection + System Loop
# ------------------------------------------------------------
def detect_frame(image_path, model):
    """Detect vehicles + occupancy from file path and visualize adjusted boxes."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None, None

    stalls = load_config()
    res = model.predict(source=img, conf=CONF, verbose=False)[0]

    # --- collect YOLO boxes ---
    boxes = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id in VEHICLE_CLASSES:
            boxes.append(tuple(map(float, b.xyxy[0].tolist())))

    occupied = {s["id"]: False for s in stalls}

    # --- visualize & apply adjusted centers ---
    for s in stalls:
        contour = np.array(s["pts"], np.int32)
        pid = s["id"]
        sx, sy, sw, sh = cv2.boundingRect(contour)

        for box in boxes[:]:
            x1, y1, x2, y2 = box

            # --- Shrink box inward to reduce overlap with nearby stalls ---
            shrink = 25
            x1 += shrink; y1 += shrink
            x2 -= shrink; y2 -= shrink

            # --- Lower center to approximate wheel/tire area ---
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2) + 25

            # --- Draw debug visualization for adjusted boxes ---
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(img, "adj", (cx - 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # --- Stall intersection checks ---
            in_poly = cv2.pointPolygonTest(contour, (cx, cy), False) >= 0
            ix1, iy1 = max(x1, sx), max(y1, sy)
            ix2, iy2 = min(x2, sx + sw), min(y2, sy + sh)
            inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            overlap = (inter_w * inter_h) / (sw * sh + 1e-6)

            # --- Occupancy decision ---
            if in_poly or overlap > 0.50:
                occupied[pid] = True
                boxes.remove(box)
                break

    overlay = draw_overlay(img, stalls, occupied, boxes)
    ts = int(time.time() * 1000)
    overlay_path = os.path.join(OVERLAYS_DIR, f"overlay_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay)
    draw_topdown_map(stalls, occupied)

    return overlay_path, occupied



def cleanup_old_files(folder, max_files=200, max_age_hours=2, keep_latest=True):
    """Delete old files by age and count."""
    if not os.path.exists(folder): return
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                   key=os.path.getmtime)
    now, cutoff = time.time(), time.time() - max_age_hours * 3600
    for f in files[:-1] if keep_latest else files:
        if os.path.getmtime(f) < cutoff:
            try: os.remove(f)
            except: pass
    while len(files) > max_files:
        oldest = files.pop(0)
        if keep_latest and oldest == files[-1]: break
        try: os.remove(oldest)
        except: pass


def main():
    ensure_dirs()
    if not os.path.exists(CONFIG):
        print("⚠️ Missing lot_config.json! Run stall_config_poly.py first.")
        return

    model = YOLO(MODEL_PATH)
    processed = set()

    print("\n🚗 Spotection System Running... (CTRL+C to stop)\n")

    try:
        while True:
            frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")), key=os.path.getmtime)
            for fp in frames:
                if fp in processed:
                    continue
                print(f"🆕 New frame detected: {fp}")
                try:
                    overlay_path, occupied = detect_frame(fp, model)
                    if overlay_path and occupied:
                        save_detection_result(
                            frame_path=fp,
                            overlay_path=overlay_path,
                            occupied_count=sum(occupied.values()),
                            free_count=len(occupied) - sum(occupied.values()),
                            stall_status=occupied
                        )
                        print(f"✅ DB updated for {os.path.basename(fp)}")
                    shutil.move(fp, os.path.join(PROCESSED_DIR, os.path.basename(fp)))
                    processed.add(fp)
                except Exception as e:
                    print(f"❌ Error processing {fp}: {e}")
            cleanup_old_files(PROCESSED_DIR, MAX_PROCESSED_FILES, MAX_AGE_HOURS, KEEP_LATEST)
            cleanup_old_files(OVERLAYS_DIR, MAX_PROCESSED_FILES, MAX_AGE_HOURS, KEEP_LATEST)
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")


if __name__ == "__main__":
    main()
