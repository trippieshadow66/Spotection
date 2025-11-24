# src/process_manager.py
import subprocess
import sys
import threading
import time

class ProcessManager:
    """
    Handles launching and stopping capture.py and detect.py
    for each lot dynamically.

    capture_processes: { lot_id: Popen }
    detect_processes:  { lot_id: Popen }
    """

    def __init__(self):
        self.capture_processes = {}
        self.detect_processes = {}

    # ------------------------------------------------------
    # Launch capture + detect for a lot
    # ------------------------------------------------------
    def start_lot(self, lot_id: int):
        """
        Start capture and detect processes for a single lot.
        """
        print(f"[ProcessManager] Starting capture + detect for lot {lot_id}")

        # Start capture
        cap = subprocess.Popen(
            [sys.executable, "-m", "src.capture", "--lot", str(lot_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.capture_processes[lot_id] = cap

        # Small delay so frames folder begins filling
        time.sleep(1.0)

        # Start detect
        det = subprocess.Popen(
            [sys.executable, "-m", "src.detect", "--lot", str(lot_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.detect_processes[lot_id] = det

        print(f"[ProcessManager] Lot {lot_id} processes started.")

    # ------------------------------------------------------
    # Stop capture + detect for a lot
    # ------------------------------------------------------
    def stop_lot(self, lot_id: int):
        """
        Kill capture & detect processes for the given lot.
        """
        print(f"[ProcessManager] Stopping processes for lot {lot_id}")

        # Stop capture
        cap = self.capture_processes.get(lot_id)
        if cap:
            try:
                cap.terminate()
                print(f"[ProcessManager] capture.py terminated for lot {lot_id}")
            except Exception as e:
                print(f"[ProcessManager] Error terminating capture for lot {lot_id}: {e}")
            self.capture_processes.pop(lot_id, None)

        # Stop detect
        det = self.detect_processes.get(lot_id)
        if det:
            try:
                det.terminate()
                print(f"[ProcessManager] detect.py terminated for lot {lot_id}")
            except Exception as e:
                print(f"[ProcessManager] Error terminating detect for lot {lot_id}: {e}")
            self.detect_processes.pop(lot_id, None)

    # ------------------------------------------------------
    # Start ALL lots on boot (optional - used by launch.py)
    # ------------------------------------------------------
    def start_all(self, lot_ids):
        """
        Optional helper used by launch.py to spin up initial lots.
        """
        for lid in lot_ids:
            try:
                self.start_lot(lid)
            except Exception as e:
                print(f"[ProcessManager] Failed to start lot {lid}: {e}")


# Create a single global instance used by app.py
process_manager = ProcessManager()
