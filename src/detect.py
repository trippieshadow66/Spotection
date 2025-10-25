import os, json, time, glob
import numpy as np, cv2
from ultralytics import YOLO
from datetime import datetime
from src.db import save_detection_result

# Optional shapely (not used in this simplified logic)
try:
    from shapely.geometry import Polygon, box
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CONFIG = "data/lot_config.json"
MODEL = "yolov8s.pt"
CONF  = 0.15              # YOLO confidence threshold
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
            "lane": s.get("lane", 1)  # <-- PRESERVE LANE HERE
        })
    return stalls


def draw_overlay(img, stalls, occupied_map):
    """Draw colored stall outlines and occupancy count box"""
    out = img.copy()
    for s in stalls:
        pid, pts = s["id"], s["pts"]
        color = (0, 0, 255) if occupied_map[pid] else (0, 200, 0)
        cv2.polylines(out, [pts], True, color, 2)
        cX = int(np.mean(pts[:, 0]))
        cY = int(np.mean(pts[:, 1]))
        cv2.putText(out, pid, (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    tot = len(stalls)
    occ = sum(occupied_map.values())
    txt = f"Occupied: {occ}/{tot}   Free: {tot - occ}"

    (text_w, text_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10),
                  (10 + text_w + 20, 10 + text_h + 10),
                  (0, 0, 0), -1)
    cv2.putText(out, txt, (20, 10 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def draw_topdown_map(stalls, occupied_map, output_dir="maps"):
    """
    Render true lot grid:
      • Lanes progress left→right by lane id.
      • Within each lane, stalls stack front→back automatically.
      • Stalls are drawn in the same overall numbering order you created them.
    """
    import cv2, os, time, numpy as np
    os.makedirs(output_dir, exist_ok=True)

    # layout constants
    stall_w, stall_h = 80, 120
    pad_x, pad_y = 30, 25
    margin_x, margin_y = 80, 80

    # find unique lanes and how many stalls each has (preserves numbering order)
    lanes = {}
    for s in sorted(stalls, key=lambda s: s["id"]):
        lid = s.get("lane", 1)
        lanes.setdefault(lid, []).append(s)

    lane_ids = sorted(lanes.keys())
    max_rows = max(len(v) for v in lanes.values())
    cols = len(lane_ids)

    canvas_w = int(margin_x*2 + cols*(stall_w+pad_x))
    canvas_h = int(margin_y*2 + max_rows*(stall_h+pad_y))
    canvas = np.ones((canvas_h, canvas_w, 3), np.uint8)*35

    # keep a counter per lane so new stalls in same lane go down one slot
    lane_row_index = {lid:0 for lid in lane_ids}

    for s in sorted(stalls, key=lambda s: s["id"]):
        pid  = s["id"]
        lid  = s.get("lane", 1)
        row  = lane_row_index[lid]
        lane_row_index[lid] += 1

        col = lane_ids.index(lid)

        x1 = int(margin_x + col*(stall_w+pad_x))
        y1 = int(margin_y + row*(stall_h+pad_y))
        x2, y2 = x1+stall_w, y1+stall_h

        color = (0,0,255) if occupied_map[pid] else (0,255,0)
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, -1)
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (255,255,255), 2)
        cv2.putText(canvas, str(pid),
                    (x1+stall_w//2-10, y1+stall_h//2+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # lane labels + legend
    for c,lid in enumerate(lane_ids):
        cv2.putText(canvas, f"Lane {lid}",
                    (int(margin_x + c*(stall_w+pad_x) + 5), 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(canvas, "Free", (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(canvas, "Occupied", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    ts = int(time.time()*1000)
    map_path = os.path.join(output_dir, f"map_{ts}.jpg")
    cv2.imwrite(map_path, canvas)
    print(f"🗺️ Wrote sequential lane map: {map_path}")
    return map_path





# ------------------------------------------------------------
# Core Detection Function  (unchanged accuracy logic)
# ------------------------------------------------------------
def detect_frame(frame_path, return_map=False):
    """Run YOLO detection on a single frame and optionally return occupancy map."""
    stalls = load_config()
    model = YOLO(MODEL)

    img = cv2.imread(frame_path)
    if img is None:
        print(" Could not read image:", frame_path)
        return None, None

    # --- Preprocess (helps dark & bright cars) ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gamma = 1.15
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, lut)
    img = np.clip(img, 0, 245).astype("uint8")

    # --- YOLO detection ---
    res = model.predict(source=img, conf=CONF, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))

    # Debug draw (optional)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 255, 0), 2)

    # --- Occupancy logic (your tuned version) ---
    occupied = {s["id"]: False for s in stalls}
    for s in stalls:
        contour = np.array(s["pts"], np.int32)
        pid = s["id"]
        sx, sy, sw, sh = cv2.boundingRect(contour)

        for box in boxes[:]:  # iterate over a copy
            x1, y1, x2, y2 = box

            shrink = 6
            x1 += shrink; y1 += shrink
            x2 -= shrink; y2 -= shrink

            # Overlap ratio
            stall_rect = (sx, sy, sx + sw, sy + sh)
            ix1, iy1 = max(x1, stall_rect[0]), max(y1, stall_rect[1])
            ix2, iy2 = min(x2, stall_rect[2]), min(y2, stall_rect[3])
            inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            stall_area = sw * sh
            overlap_ratio = inter_area / (stall_area + 1e-6)

            # Center test
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            in_poly = cv2.pointPolygonTest(contour, (cx, cy), False) >= 0

            if in_poly or overlap_ratio > 0.25:
                occupied[pid] = True
                boxes.remove(box)  # one detection can fill only one stall
                break

    # --- Draw overlay and save ---
    overlay = draw_overlay(img, stalls, occupied)
    os.makedirs("overlays", exist_ok=True)
    ts = int(time.time() * 1000)
    overlay_path = os.path.join("overlays", f"overlay_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay)
    print(" Wrote overlay:", overlay_path)

    # --- NEW: also generate schematic map view ---
    draw_topdown_map(stalls, occupied)

    if return_map:
        return overlay_path, occupied

    # Single-frame DB save (if not batch mode)
    save_detection_result(
        frame_path=frame_path,
        overlay_path=overlay_path,
        occupied_count=sum(occupied.values()),
        free_count=len(stalls) - sum(occupied.values()),
        stall_status=occupied
    )

    return overlay_path, occupied


# ------------------------------------------------------------
# Batch Mode (same logic, but averaged before DB commit)
# ------------------------------------------------------------
def main():
    frames = sorted([p for p in glob.glob("data/frames/*")
                     if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not frames:
        print(" No frames found in data/frames.")
        return

    # --- Take last 3 frames for voting ---
    recent = frames[-3:]
    batch_results = []

    for fp in recent:
        _, occ_map = detect_frame(fp, return_map=True)
        if occ_map:
            batch_results.append(occ_map)

    # --- Majority vote per stall ---
    stalls = load_config()
    final_status = {}
    for s in stalls:
        pid = s["id"]
        vals = [r.get(pid, False) for r in batch_results]
        occ_count = sum(vals)
        final_status[pid] = True if occ_count >= 2 else False

    occ_total = sum(final_status.values())
    free_total = len(final_status) - occ_total
    print(f" Batch summary → Occupied: {occ_total} | Free: {free_total}")

    # --- Save batch-averaged result to DB ---
    save_detection_result(
        frame_path="batch",
        overlay_path="",
        occupied_count=occ_total,
        free_count=free_total,
        stall_status=final_status
    )

    # --- NEW: also draw schematic map for batch summary ---
    draw_topdown_map(stalls, final_status)


if __name__ == "__main__":
    main()
