import os, json, glob, cv2, numpy as np, time, argparse

# Lot-specific live stream URLs
LIVE_STREAMS = {
    1: "https://taco-about-python.com/video_feed",
    2: "http://170.249.152.2:8080/cgi-bin/viewer/video.jpg",
}

# Per-lot flip rules
NEEDS_FLIP = {
    1: True,
    2: False,
}

WIN = "Stall Config (n=new stall, u=undo, r=remove, s=save, q=quit)"

# ------------------------------------------------------------
# FOLDER CREATION
# ------------------------------------------------------------
def ensure_lot_dirs(lot_id: int):
    base_dir = os.path.join("data", f"lot{lot_id}")
    frames_dir = os.path.join(base_dir, "frames")

    os.makedirs(frames_dir, exist_ok=True)
    print(f"📁 Ensured: {base_dir}/ + frames/")
    return base_dir, frames_dir

# ------------------------------------------------------------
# GET BASE IMAGE
# ------------------------------------------------------------
def get_base_image(lot_id: int, frames_dir: str):
    files = sorted([
        p for p in glob.glob(os.path.join(frames_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if files:
        img = cv2.imread(files[-1])
        if img is not None:
            print("📸 Using existing snapshot.")
            return img

    print("⚠️ No frames. Warming camera...")

    url = LIVE_STREAMS[lot_id]
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera.")

    good = None
    start = time.time()
    while time.time() - start < 2.5:
        ok, frame = cap.read()
        if not ok:
            continue
        if NEEDS_FLIP.get(lot_id, False):
            frame = cv2.flip(frame, 0)

        if frame.mean() > 5:
            good = frame.copy()

    cap.release()
    if good is None:
        raise RuntimeError("❌ Camera warm-up failed.")

    snap = os.path.join(frames_dir, "live_snapshot.jpg")
    cv2.imwrite(snap, good)
    print(f"✅ Saved warm-up snapshot: {snap}")
    return good

# ------------------------------------------------------------
# DRAWING
# ------------------------------------------------------------
def draw_poly(img, pts, color=(0,255,255)):
    pts = [tuple(p) for p in pts]
    for p in pts:
        cv2.circle(img, p, 3, (0,255,0), -1)
    if len(pts) > 1:
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], color, 2)
    if len(pts) > 2:
        cv2.line(img, pts[-1], pts[0], color, 1)

def point_in_poly(point, poly):
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside = not inside
    return inside

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", type=int, required=True)
    args = parser.parse_args()

    lot_id = args.lot

    # Setup
    base_dir, frames_dir = ensure_lot_dirs(lot_id)
    CONFIG_PATH = os.path.join(base_dir, "lot_config.json")

    base = get_base_image(lot_id, frames_dir)
    view = base.copy()

    stalls = []
    current = []
    remove_mode = False

    # -------------------------
    # REDRAW FIXED
    # -------------------------
    def redraw(highlight_remove=False):
        canvas = base.copy()

        # draw saved stalls
        for s in stalls:
            c = (0,0,255) if highlight_remove else (0,255,255)
            draw_poly(canvas, s["points"], c)
            cx, cy = np.mean(np.array(s["points"]), axis=0).astype(int)
            cv2.putText(canvas, str(s["id"]), (cx-8, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # draw current polygon
        draw_poly(canvas, current, (255,200,0))

        return canvas

    # -------------------------
    # MOUSE CALLBACK
    # -------------------------
    def on_mouse(event, x, y, flags, param):
        nonlocal view, current, stalls, remove_mode

        if event == cv2.EVENT_LBUTTONDOWN:
            if remove_mode:
                # remove stall
                for s in stalls[:]:
                    if point_in_poly((x,y), s["points"]):
                        print(f"🗑️ Removed stall {s['id']}")
                        stalls.remove(s)
                        # renumber
                        for i, st in enumerate(stalls, 1):
                            st["id"] = i
                        view = redraw(True)
                        return
            else:
                # add point
                current.append([x, y])
                view = redraw(remove_mode)

    # -------------------------
    # WINDOW SETUP
    # -------------------------
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(base.shape[1], 1280), min(base.shape[0], 720))
    cv2.setMouseCallback(WIN, on_mouse)

    print("\nINSTRUCTIONS:")
    print("  Click = add point")
    print("  n = finish stall and choose lane")
    print("  u = undo last point")
    print("  r = toggle remove mode")
    print("  s = save and exit")
    print("  q = quit without saving\n")

    # -------------------------
    # MAIN LOOP
    # -------------------------
    while True:
        cv2.imshow(WIN, view)
        k = cv2.waitKey(50) & 0xFF

        if k == ord("u"):
            if current:
                current.pop()
                print("↩️ Undo last point.")
            view = redraw(remove_mode)

        elif k == ord("r"):
            remove_mode = not remove_mode
            print(f"🧹 Remove mode {'ON' if remove_mode else 'OFF'}")
            view = redraw(remove_mode)

        elif k == ord("n"):
            if len(current) < 3:
                print("⚠️ Need at least 3 points.")
                continue

            sid = len(stalls)+1
            print(f"➡️ Stall {sid} complete. Press lane (1–9).")

            lane = None
            while lane is None:
                lk = cv2.waitKey(0) & 0xFF
                if ord("1") <= lk <= ord("9"):
                    lane = int(chr(lk))

            stalls.append({"id": sid, "lane": lane, "points": current.copy()})
            current.clear()
            view = redraw(remove_mode)

        elif k == ord("s"):
            # save and exit
            if len(current) >= 3:
                sid = len(stalls)+1
                print(f"➡️ Completing stall {sid}. Choose lane (1–9).")
                lane = None
                while lane is None:
                    lk = cv2.waitKey(0) & 0xFF
                    if ord("1") <= lk <= ord("9"):
                        lane = int(chr(lk))
                stalls.append({"id": sid, "lane": lane, "points": current.copy()})

            cfg = {"stalls": stalls}
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"💾 Saved config → {CONFIG_PATH}")
            break

        elif k == ord("q"):
            print("❌ Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
