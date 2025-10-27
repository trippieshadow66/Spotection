import os, json, time, glob, cv2, numpy as np
from datetime import datetime
from ultralytics import YOLO
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


def draw_overlay(img, stalls, occupied_map):
    """Draw colored stall outlines and occupancy count box"""
    out = img.copy()
    for s in stalls:
        pid, pts = s["id"], s["pts"]
        color = (0, 0, 255) if occupied_map[pid] else (0, 200, 0)
        cv2.polylines(out, [pts], True, color, 2)
        cX, cY = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        cv2.putText(out, pid, (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    tot, occ = len(stalls), sum(occupied_map.values())
    txt = f"Occupied: {occ}/{tot}   Free: {tot - occ}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 20, 10 + th + 10), (0, 0, 0), -1)
    cv2.putText(out, txt, (20, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def draw_topdown_map(stalls, occupied_map, output_dir="maps"):
    """
    Draw a top-down schematic where:
      - Lanes progress left → right by lane id.
      - Within each lane, stalls go front → back (closest to camera first).
    """
    import os, time, cv2, numpy as np
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

    # --- Sort lanes left→right and stalls front→back (FLIPPED order) ---
    lane_ids = sorted(lanes.keys())
    for lid in lane_ids:
        # Smaller Y = farther away, larger Y = closer → reverse sort flips front/back
        lanes[lid].sort(key=lambda s: np.mean(s["pts"][:, 1]), reverse=False)

    # --- Determine grid size ---
    max_rows = max(len(v) for v in lanes.values())
    cols = len(lane_ids)
    canvas_w = int(margin_x * 2 + cols * (stall_w + pad_x))
    canvas_h = int(margin_y * 2 + max_rows * (stall_h + pad_y))
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, np.uint8)

    # --- Draw each lane column ---
    for c, lid in enumerate(lane_ids):
        stalls_in_lane = lanes[lid]
        # Draw from top down, so stall[0] (closest to camera) appears at top
        for r, s in enumerate(stalls_in_lane):
            pid = s["id"]
            # Flip the order vertically
            

            color = (0, 0, 255) if occupied_map.get(pid) else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(canvas, str(pid),
                        (x1 + stall_w // 2 - 10, y1 + stall_h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Lane header label (top of each column) ---
        label_x = int(margin_x + c * (stall_w + pad_x) + 5)
        cv2.putText(canvas, f"Lane {lid}", (label_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Legend ---
    cv2.putText(canvas, "Free", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, "Occupied", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    ts = int(time.time() * 1000)
    map_path = os.path.join(output_dir, f"map_{ts}.jpg")
    cv2.imwrite(map_path, canvas)
    print(f"✅ Wrote front-to-back flipped map: {map_path}")
    return map_path



# ------------------------------------------------------------
# Core Detection Function (now compatible with return_map)
# ------------------------------------------------------------
def detect_frame(frame_path, return_map=False):
    """Run YOLO detection on a saved frame path, optionally returning occupancy map."""
    stalls = load_config()
    model = YOLO(MODEL)
    img = cv2.imread(frame_path)
    if img is None:
        print("⚠️ Could not read image:", frame_path)
        return None, None

    # --- Preprocessing ---
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

    res = model.predict(source=img, conf=CONF, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))

    occupied = {s["id"]: False for s in stalls}
    for s in stalls:
        contour = np.array(s["pts"], np.int32)
        pid = s["id"]
        sx, sy, sw, sh = cv2.boundingRect(contour)

        for box in boxes[:]:
            x1, y1, x2, y2 = box
            shrink = 20
            x1 += shrink; y1 += shrink
            x2 -= shrink; y2 -= shrink
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2) + 15
            ix1, iy1 = max(x1, sx), max(y1, sy)
            ix2, iy2 = min(x2, sx + sw), min(y2, sy + sh)
            inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            overlap_ratio = (inter_w * inter_h) / (sw * sh + 1e-6)
            in_poly = cv2.pointPolygonTest(contour, (cx, cy), False) >= 0

            if in_poly or overlap_ratio > 0.25:
                occupied[pid] = True
                boxes.remove(box)
                break

    overlay = draw_overlay(img, stalls, occupied)
    os.makedirs("overlays", exist_ok=True)
    ts = int(time.time() * 1000)
    overlay_path = os.path.join("overlays", f"overlay_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay)
    draw_topdown_map(stalls, occupied)

    if return_map:
        return overlay_path, occupied

    save_detection_result(
        frame_path=frame_path,
        overlay_path=overlay_path,
        occupied_count=sum(occupied.values()),
        free_count=len(stalls) - sum(occupied.values()),
        stall_status=occupied
    )
    return overlay_path, occupied


# ------------------------------------------------------------
# Batch Mode (unchanged logic)
# ------------------------------------------------------------
def main():
    frames = sorted([p for p in glob.glob("data/frames/*")
                     if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not frames:
        print("No frames found in data/frames.")
        return

    recent = frames[-3:]
    results = []
    for fp in recent:
        _, occ = detect_frame(fp, return_map=True)
        if occ:
            results.append(occ)

    stalls = load_config()
    final = {}
    for s in stalls:
        pid = s["id"]
        vals = [r.get(pid, False) for r in results]
        final[pid] = sum(vals) >= 2

    occ_total = sum(final.values())
    free_total = len(final) - occ_total
    print(f"Batch summary → Occupied: {occ_total} | Free: {free_total}")
    save_detection_result(
        frame_path="batch",
        overlay_path="",
        occupied_count=occ_total,
        free_count=free_total,
        stall_status=final
    )
    draw_topdown_map(stalls, final)


if __name__ == "__main__":
    main()
