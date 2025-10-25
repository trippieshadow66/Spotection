import json, os, glob, cv2, numpy as np

WIN = "Stall Config (n=new stall, u=undo point, s=save, q=quit)"
CONFIG_PATH = "data/lot_config.json"


def latest_frame():
    files = sorted([
        p for p in glob.glob("data/frames/*")
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return files[-1] if files else None


def draw_poly(img, poly, color=(0, 255, 255)):
    """Draw a polygon being defined."""
    if len(poly) == 0:
        return
    for i, p in enumerate(poly):
        cv2.circle(img, tuple(p), 3, (0, 255, 0), -1)
        if i > 0:
            cv2.line(img, tuple(poly[i - 1]), tuple(p), color, 2)
    if len(poly) > 2:
        cv2.line(img, tuple(poly[-1]), tuple(poly[0]), color, 1)


def main():
    os.makedirs("data", exist_ok=True)
    img_path = latest_frame()
    if not img_path:
        print("No images in frames/.")
        return

    base = cv2.imread(img_path)
    if base is None:
        print("Cannot read", img_path)
        return

    view = base.copy()
    stalls, current = [], []

    def redraw():
        """Refresh window view with all polygons."""
        v = base.copy()
        for s in stalls:
            draw_poly(v, [tuple(p) for p in s["points"]])
        draw_poly(v, [tuple(p) for p in current], (255, 200, 0))
        cv2.imshow(WIN, v)
        return v

    def on_mouse(event, x, y, flags, param):
        nonlocal view, current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])
            view = redraw()

    print("\nINSTRUCTIONS:")
    print("- Left-click to add polygon points around ONE stall.")
    print("- Press 'n' to finish that stall and assign lane (1–9).")
    print("- Press 'u' to undo last point.")
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
            view = redraw()

        elif k == ord('n'):
            if len(current) < 3:
                print("Need ≥3 points to form a stall polygon.")
                continue

            stall_id = len(stalls) + 1
            print(f"Stall {stall_id} complete. Press a number key (1–9) to assign lane.")
            cv2.imshow(WIN, view)

            lane_key = None
            while lane_key is None:
                lk = cv2.waitKey(0) & 0xFF
                if ord('1') <= lk <= ord('9'):
                    lane_key = int(chr(lk))
                    print(f" → Lane {lane_key} assigned.")
                elif lk == 27 or lk == ord('q'):
                    print("Cancelled stall assignment.")
                    break

            if lane_key is not None:
                stalls.append({
                    "id": stall_id,
                    "lane": lane_key,
                    "points": current.copy()
                })
                current = []
                view = redraw()

        elif k == ord('s'):
            # finish last polygon if mid-drawing
            if len(current) >= 3:
                stall_id = len(stalls) + 1
                print(f"Stall {stall_id} complete. Press 1–9 for lane:")
                lane_key = None
                while lane_key is None:
                    lk = cv2.waitKey(0) & 0xFF
                    if ord('1') <= lk <= ord('9'):
                        lane_key = int(chr(lk))
                        print(f" → Lane {lane_key} assigned.")
                        break
                stalls.append({
                    "id": stall_id,
                    "lane": lane_key,
                    "points": current.copy()
                })

            cfg = {"image": os.path.basename(img_path), "stalls": stalls}
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print("✅ Saved", CONFIG_PATH)
            break

        elif k == ord('q') or k == 27:
            print("❌ Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
