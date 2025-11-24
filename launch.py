import subprocess, time, sys
from src.db import init_db, get_all_lots


def main():
    print("🚀 Starting Spotection Multi-Lot System\n")

    print("📊 Initializing DB…")
    init_db()

    lot_rows = get_all_lots()
    LOTS = [row["id"] for row in lot_rows]
    if not LOTS:
        print("⚠️ No lots defined in DB. Create one via the admin UI first.")
        return

    print(f"📋 Lots to start: {LOTS}")

    cap_processes = []
    detect_processes = []

    for lot in LOTS:
        print(f"📷 Starting capture for lot {lot}")
        p = subprocess.Popen([sys.executable, "-m", "src.capture", "--lot", str(lot)])
        cap_processes.append(p)

    time.sleep(5)

    for lot in LOTS:
        print(f"🧠 Starting detection for lot {lot}")
        p = subprocess.Popen([sys.executable, "-m", "src.detect", "--lot", str(lot)])
        detect_processes.append(p)

    print("🌍 Starting Flask web dashboard… http://localhost:5000")
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("🛑 Shutting down…")

    for p in cap_processes + detect_processes:
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
