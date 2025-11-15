import subprocess
import time
import sys
import os

def main():
    print("=" * 50)
    print("🚀 STARTING SPOTECTION FULL SYSTEM")
    print("=" * 50)
    
    # Step 1: Initialize database
    print("📊 Step 1: Initializing database...")
    try:
        subprocess.run([sys.executable, "-m", "src.db"], check=True)
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return
    
    # Step 2: Start camera capture system (in background)
    print("📷 Step 2: Starting camera capture...")
    try:
        # Start capture as a separate process
        capture_process = subprocess.Popen([sys.executable, "-m", "src.capture"])
        print("✅ Camera capture started in background")
    except Exception as e:
        print(f"❌ Camera capture failed: {e}")
        return
    
    # Step 3: Wait a moment for systems to initialize
    print("⏳ Step 3: Waiting for systems to initialize...")
    time.sleep(5)
    
    # Step 4: Start the web dashboard
    print("🌐 Step 4: Starting web dashboard...")
    print("=" * 50)
    print("Your parking system will be available at: http://localhost:5000")
    print("=" * 50)
    
    # This will keep running until you press Ctrl+C
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Spotection system...")
        capture_process.terminate()

if __name__ == "__main__":
    main()