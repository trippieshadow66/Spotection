import subprocess, sys, time, os

class ProcessManager:
    def __init__(self):
        self.capture_processes = {}
        self.detect_processes = {}

    def ensure_folders(self, lot_id):
        base = f"data/lot{lot_id}"
        for sub in ("frames", "overlays", "maps"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    def start_lot(self, lot_id):
        print(f"[PM] Starting lot {lot_id}")

        # Auto-create folders for migrated DBs
        self.ensure_folders(lot_id)

        # Start capture
        cap = subprocess.Popen(
            [sys.executable, "-m", "capture", "--lot", str(lot_id)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.capture_processes[lot_id] = cap

        time.sleep(1)

        det = subprocess.Popen(
            [sys.executable, "-m", "src.detect", "--lot", str(lot_id)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.detect_processes[lot_id] = det

        print(f"[PM] Lot {lot_id} started.")

    def stop_lot(self, lot_id):
        print(f"[PM] Stopping {lot_id}")

        if lot_id in self.capture_processes:
            try:
                self.capture_processes[lot_id].terminate()
            except: pass
            self.capture_processes.pop(lot_id, None)

        if lot_id in self.detect_processes:
            try:
                self.detect_processes[lot_id].terminate()
            except: pass
            self.detect_processes.pop(lot_id, None)

        print(f"[PM] Lot {lot_id} stopped.")

    def start_all(self, lot_ids):
        for lid in lot_ids:
            try:
                self.start_lot(lid)
            except Exception as e:
                print(f"[PM] Failed starting {lid}: {e}")


process_manager = ProcessManager()
