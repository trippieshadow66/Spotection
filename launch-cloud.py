import subprocess
import sys

def main():
    print("🚀 Starting Spotection (Cloud Version)")
    
    # Initialize database only (skip camera on cloud)
    print("📊 Initializing database...")
    subprocess.run([sys.executable, "-m", "src.db"])
    
    # Start web app only
    print("🌐 Starting web dashboard...")
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()