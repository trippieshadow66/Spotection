import cv2, numpy as np, os, sys, glob

POINT_RADIUS = 5
POINT_COLOR = (0, 255, 255)  # yellow
LINE_COLOR = (0, 255, 0)     # green
WIN = "Spotection Homography Calibration"

def order_points(pts):
    # pts: 4x2
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def distance(a, b):
    return np.linalg.norm(a - b)

def compute_output_size(ordered):
    (tl, tr, br, bl) = ordered
    widthA = distance(br, bl)
    widthB = distance(tr, tl)
    heightA = distance(tr, br)
    heightB = distance(tl, bl)
    W = int(max(widthA, widthB))
    H = int(max(heightA, heightB))
    # avoid zero sizes
    return max(W, 100), max(H, 100)

def find_latest_frame():
    files = sorted(glob.glob(os.path.join("frames", "*.*")))
    return files[-1] if files else None

def main():
    # pick image: arg or latest in frames/
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = find_latest_frame()
    if not img_path or not os.path.exists(img_path):
        print("No calibration image found. Provide a path or put images in frames/.")
        return

    img = cv2.imread(img_path)
    clone = img.copy()
    h, w = img.shape[:2]
    clicks = []

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, img
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(img, (x, y), POINT_RADIUS, POINT_COLOR, -1)
            if len(clicks) > 1:
                cv2.line(img, clicks[-2], clicks[-1], LINE_COLOR, 2)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(WIN, on_mouse)

    print("\nINSTRUCTIONS")
    print("1) Click the FOUR corners of the lot/ROI (roughly TL, TR, BR, BL).")
    print("2) Press 'r' to reset clicks, 'p' to preview warp, 's' to save, 'q' to quit.\n")

    preview = None
    Hmat = None
    outW = outH = None

    while True:
        cv2.imshow(WIN, img if preview is None else preview)
        k = cv2.waitKey(20) & 0xFF

        if k == ord('r'):
            img = clone.copy()
            clicks = []
            preview = None
            Hmat = None
            print("Reset.")

        elif k == ord('p'):
            if len(clicks) != 4:
                print("Need 4 points before preview.")
                continue
            src = order_points(clicks)
            outW, outH = compute_output_size(src)
            dst = np.array([[0,0],[outW,0],[outW,outH],[0,outH]], dtype=np.float32)
            Hmat = cv2.getPerspectiveTransform(src, dst)
            preview = cv2.warpPerspective(clone, Hmat, (outW, outH))
            print(f"Preview: {outW}x{outH}")

        elif k == ord('s'):
            if Hmat is None:
                print("Preview first ('p') so we can compute H.")
                continue
            os.makedirs("data", exist_ok=True)
            np.savez("data/homography.npz",
                     H=Hmat, out_width=outW, out_height=outH,
                     src_points=order_points(clicks),
                     dst_points=np.array([[0,0],[outW,0],[outW,outH],[0,outH]], dtype=np.float32),
                     source_image=os.path.basename(img_path))
            if preview is not None:
                cv2.imwrite("data/warped_preview.jpg", preview)
            print("Saved: data/homography.npz and data/warped_preview.jpg")
            break

        elif k == ord('q') or k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
