import cv2, os, time
from datetime import datetime

# Ensure output folder exists
os.makedirs("data/frames", exist_ok=True)

# Try backend with my camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(" Could not open camera. Try changing VideoCapture(0) → (1) or (2).")

print(" Camera opened. Capturing frames... Press CTRL+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" No frame received. Exiting.")
            break

        # Save snapshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/frames/frame_{ts}.jpg"
        cv2.imwrite(filename, frame)
        print("📸 Saved:", filename)

        # Show live preview
        cv2.imshow("Spotection Capture", frame)

        # Wait ~2s, press q to quit
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
