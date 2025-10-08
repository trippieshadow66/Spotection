import os, json, time, glob
import numpy as np, cv2
from ultralytics import YOLO

# try shapely (nice overlap area). If not available, we'll do center-in-polygon fallback.
try:
    from shapely.geometry import Polygon, box
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

CONFIG = "data/lot_config.json"
MODEL = "yolov8s.pt"   # small & fast; try 'yolov8s.pt' for better accuracy
CONF  = 0.15           # YOLO confidence threshold
IOU_THRESH = 0.3      # fraction of stall area overlapped by a box to call 'occupied'

# COCO class ids to consider as "vehicles"
VEHICLE_CLASSES = {2, 7}   # car, truck

def load_config():
    with open(CONFIG, "r") as f:
        cfg = json.load(f)
    stalls = []
    for s in cfg["stalls"]:
        pts = np.array(s["points"], dtype=np.int32)
        stalls.append({"id": str(s["id"]), "pts": pts})
    return stalls

def point_in_poly_center(contour, x1,y1,x2,y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cv2.pointPolygonTest(contour, (cx, cy), False) >= 0

def draw_overlay(img, stalls, occupied_map):
    out = img.copy()
    for s in stalls:
        pid, pts = s["id"], s["pts"]
        color = (0,0,255) if occupied_map[pid] else (0,200,0)
        cv2.polylines(out, [pts], True, color, 2)
        cX = int(np.mean(pts[:,0])); cY = int(np.mean(pts[:,1]))
        cv2.putText(out, pid, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # summary
    tot = len(stalls)
    occ = sum(occupied_map.values())
    txt = f"Occupied: {occ}/{tot}   Free: {tot-occ}"
    cv2.rectangle(out, (10,10), (10+320, 10+34), (0,0,0), -1)
    cv2.putText(out, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return out

def main():
    os.makedirs("overlays", exist_ok=True)
    stalls = load_config()
    model = YOLO(MODEL)

    frames = sorted([p for p in glob.glob("data/frames/*") if p.lower().endswith((".jpg",".jpeg",".png"))])
    if not frames:
        print("No frames/ images found."); return

    # test last 3 frames (change to [-1:] for just the newest)
    for fp in frames[-3:]:
        img = cv2.imread(fp)
        if img is None: continue
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        res = model.predict(source=img, conf=CONF, verbose=False)[0]
        # collect vehicle boxes
        boxes = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

        #  Draw YOLO detection boxes in light blue (for debugging)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        occupied = {s["id"]: False for s in stalls}

        if SHAPELY_OK:
            # precise polygon overlap
            stall_polys = {s["id"]: Polygon([(int(x),int(y)) for x,y in s["pts"]]) for s in stalls}
            for pid, poly in stall_polys.items():
                for (x1,y1,x2,y2) in boxes:
                    inter = poly.intersection(box(x1,y1,x2,y2))
                    if not inter.is_empty and (inter.area / max(poly.area,1e-6)) >= IOU_THRESH:
                        occupied[pid] = True
                        break
        else:
            # fallback: center of the box inside polygon
            for s in stalls:
                contour = s["pts"]
                pid = s["id"]
                for (x1,y1,x2,y2) in boxes:
                    if point_in_poly_center(contour, x1,y1,x2,y2):
                        occupied[pid] = True
                        break

        overlay = draw_overlay(img, stalls, occupied)
        ts = int(time.time()*1000)
        outp = os.path.join("overlays", f"overlay_{ts}.jpg")
        cv2.imwrite(outp, overlay)
        print("Wrote:", outp)

if __name__ == "__main__":
    main()
