import os, json, time, glob, cv2, numpy as np
from ultralytics import YOLO
from datetime import datetime
from src.db import save_detection_result

CONFIG = "data/lot_config.json"
MODEL = "yolov8s.pt"
CONF = 0.15
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def load_config():
    """Load stall polygons (and lane numbers) from lot_config.json"""
    if not os.path.exists(CONFIG):
        raise FileNotFoundError("Missing data/lot_config.json. Run stall_config_poly.py first.")
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
    """Draw colored stall outlines, YOLO boxes, center dots, and occupancy text."""
    out = img.copy()

    # --- YOLO detection visuals ---
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(out, "C", (cx - 5, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- Stall polygons ---
    for s in stalls:
        pid, pts = s["id"], s["pts"]
        color = (0, 0, 255) if occupied_map.get(pid) else (0, 200, 0)
        cv2.polylines(out, [pts], True, color, 2)
        cX, cY = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        cv2.putText(out, pid, (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Occupancy summary ---
    tot, occ = len(stalls), sum(occupied_map.values())
    txt = f"Occupied: {occ}/{tot}   Free: {tot - occ}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 20, 10 + th + 10), (0, 0, 0), -1)
    cv2.putText(out, txt, (20, 10 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def draw_topdown_map(stalls, occupied_map, output_dir="maps"):
    """
    Draw schematic preserving real lot layout:
      • Lanes drawn left → right.
      • Within each lane, stalls drawn top → bottom (back → front).
      • Each lane column represents one physical lane.
    """
    import cv2, numpy as np, os, time
    os.makedirs(output_dir, exist_ok=True)

    # --- Layout constants ---
    stall_w, stall_h = 120, 80
    pad_x, pad_y = 30, 25
    margin_x, margin_y = 80, 80
    bg_color = (35, 35, 35)

    # --- Group stalls by lane ---
    lanes = {}
    for s in stalls:
        lid = s.get("lane", 1)
        lanes.setdefault(lid, []).append(s)

    # --- Sort lanes left→right, stalls back→front (top→bottom) ---
    lane_ids = sorted(lanes.keys())
    for lid in lane_ids:
        # sort by Y centroid ascending (small Y = farther back)
        lanes[lid].sort(key=lambda s: np.mean(s["pts"][:, 1]))

    # --- Determine canvas size ---
    max_rows = max(len(v) for v in lanes.values())
    cols = len(lane_ids)
    canvas_w = int(margin_x * 2 + cols * (stall_w + pad_x))
    canvas_h = int(margin_y * 2 + max_rows * (stall_h + pad_y))
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, np.uint8)

    # --- Draw each lane column ---
    for c, lid in enumerate(lane_ids):
        stalls_in_lane = lanes[lid]
        for r, s in enumerate(stalls_in_lane):
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

        # --- Lane header label ---
        label_x = int(margin_x + c * (stall_w + pad_x) + 5)
        cv2.putText(canvas, f"Lane {lid}", (label_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Legend + Orientation ---
    cv2.putText(canvas, "Back of Lot ↑", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, "Front of Lot ↓", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    ts = int(time.time() * 1000)
    map_path = os.path.join(output_dir, f"map_{ts}.jpg")
    cv2.imwrite(map_path, canvas)
    print(f"✅ Wrote map preserving lot layout: {map_path}")
    return map_path



# ------------------------------------------------------------
# Core Detection (used in both run_system and app.py)
# ------------------------------------------------------------
def detect_frame(image, model=None, return_map=False):
    """Run YOLO detection directly on a frame array (used for live feed)."""
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            print(f"⚠️ Could not read image file: {image}")
            return None, None
    else:
        img = image.copy()
    stalls = load_config()
    if model is None:
        model = YOLO(MODEL)
    res = model.predict(source=image, conf=CONF, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    # --- Occupancy logic ---
    occupied = {s["id"]: False for s in stalls}
    for s in stalls:
        contour = np.array(s["pts"], np.int32)
        pid = s["id"]
        sx, sy, sw, sh = cv2.boundingRect(contour)
        for box in boxes[:]:
            x1, y1, x2, y2 = box
            shrink = 15
            x1 += shrink; y1 += shrink; x2 -= shrink; y2 -= shrink
            cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2) + 25
            in_poly = cv2.pointPolygonTest(contour, (cx, cy), False) >= 0
            ix1, iy1 = max(x1, sx), max(y1, sy)
            ix2, iy2 = min(x2, sx + sw), min(y2, sy + sh)
            inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            overlap = (inter_w * inter_h) / (sw * sh + 1e-6)
            if in_poly or overlap > 0.25:
                occupied[pid] = True
                boxes.remove(box)
                break
    overlay = draw_overlay(image, stalls, occupied, boxes)
    draw_topdown_map(stalls, occupied)
    return overlay, occupied
