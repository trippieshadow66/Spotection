import os, json, glob, cv2, numpy as np

WIN = "Stall Config (n=new stall, u=undo point, s=save, q=quit)"
CONFIG_PATH = "data/lot_config.json"
LIVE_STREAM_URL = "https://taco-about-python.com/video_feed"

def get_base_image():
    """Try latest frame; if none, grab a live snapshot."""
    files = sorted([
        p for p in glob.glob("data/frames/*")
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if files:
        img = cv2.imread(files[-1])
        
        return img
    print("⚠️ No frames found, grabbing from live feed...")
    cap = cv2.VideoCapture(LIVE_STREAM_URL)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read from live stream.")
    os.makedirs("data/frames", exist_ok=True)
    snap_path = "data/frames/live_snapshot.jpg"
    
    cv2.imwrite(snap_path, frame)
    print(f"✅ Saved snapshot → {snap_path}")
    return frame


def draw_poly(img, poly, color=(0, 255, 255)):
    if len(poly) == 0: return
    for i, p in enumerate(poly):
        cv2.circle(img, tuple(p), 3, (0, 255, 0), -1)
        if i > 0: cv2.line(img, tuple(poly[i - 1]), tuple(p), color, 2)
    if len(poly) > 2:
        cv2.line(img, tuple(poly[-1]), tuple(poly[0]), color, 1)


def main():
    os.makedirs("data", exist_ok=True)
    base = get_base_image()
    if base is None:
        print("❌ Could not load image.")
        return

    view = base.copy()
    stalls, current = [], []
    remove_mode = False

    def redraw(highlight_remove=False):
        v = base.copy()
        for s in stalls:
            color = (0, 0, 255) if highlight_remove else (0, 255, 255)
            draw_poly(v, [tuple(p) for p in s["points"]], color)
            cX, cY = np.mean(np.array(s["points"]), axis=0).astype(int)
            cv2.putText(v, str(s["id"]), (cX - 8, cY - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        draw_poly(v, [tuple(p) for p in current], (255, 200, 0))
        cv2.imshow(WIN, v)
        return v

    def on_mouse(event, x, y, flags, param):
        nonlocal view, current, stalls, remove_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if remove_mode:
                for s in stalls[:]:
                    if point_in_poly((x, y), s["points"]):
                        print(f"🗑️  Removed stall {s['id']} (Lane {s['lane']}).")
                        stalls.remove(s)
                        # re-number IDs so they stay sequential
                        for i, st in enumerate(stalls, 1):
                            st["id"] = i
                        redraw(True)
                        return
            else:
                current.append([x, y])
                view = redraw()

    print("\nINSTRUCTIONS:")
    print("- Left-click to add polygon points around ONE stall.")
    print("- Press 'n' to finish that stall and assign lane (1–9).")
    print("- Press 'u' to undo last point.")
    print("- Press 'r' to toggle REMOVE mode (click inside a stall to delete).")
    print("- Press 's' to save and exit, or 'q' to quit without saving.\n")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(base.shape[1], 1280), min(base.shape[0], 720))
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        cv2.imshow(WIN, view)
        k = cv2.waitKey(25) & 0xFF

        if k == ord('u'):
            if current:
                current.pop()
            view = redraw(remove_mode)

        elif k == ord('r'):
            remove_mode = not remove_mode
            state = "ON" if remove_mode else "OFF"
            print(f"🧹 Remove mode {state}. Click inside a stall to delete.")
            view = redraw(highlight_remove=remove_mode)

        elif k == ord('n'):
            if len(current) < 3:
                print("Need ≥3 points to form a stall polygon.")
                continue
            stall_id = len(stalls) + 1
            print(f"Stall {stall_id} complete. Press 1–9 for lane assignment.")
            lane = None
            while lane is None:
                lk = cv2.waitKey(0) & 0xFF
                if ord('1') <= lk <= ord('9'):
                    lane = int(chr(lk))
                    print(f" → Lane {lane} assigned.")
            stalls.append({"id": stall_id, "lane": lane, "points": current.copy()})
            current = []
            view = redraw(remove_mode)

        elif k == ord('s'):
            if len(current) >= 3:
                stall_id = len(stalls) + 1
                print(f"Stall {stall_id} complete. Press 1–9 for lane assignment.")
                lane = None
                while lane is None:
                    lk = cv2.waitKey(0) & 0xFF
                    if ord('1') <= lk <= ord('9'):
                        lane = int(chr(lk))
                        print(f" → Lane {lane} assigned.")
                        break
                stalls.append({"id": stall_id, "lane": lane, "points": current.copy()})

            cfg = {"stalls": stalls}
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print("✅ Saved", CONFIG_PATH)
            break

        elif k in (ord('q'), 27):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
