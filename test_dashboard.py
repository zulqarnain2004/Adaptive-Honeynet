#!/usr/bin/env python3
"""
Test script to verify dashboard works
"""

import subprocess
import sys
import time
import webbrowser
import socket

def check_port(port=8501):
    """Check if port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def kill_existing_streamlit():
    """Kill any existing Streamlit processes"""
    print("Checking for existing Streamlit processes...")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "streamlit.exe"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "streamlit"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        print("✅ Cleaned up old processes")
    except:
        print("⚠️ Could not clean up processes")

def test_dashboard():
    """Test the dashboard"""
    print("\n" + "="*60)
    print("🛡️ Testing Adaptive Deception-Mesh Dashboard")
    print("="*60)
    
    # Check if dashboard.py exists
    import os
    if not os.path.exists("dashboard.py"):
        print("❌ Error: dashboard.py not found!")
        print("Make sure you're in the project directory.")
        return False
    
    # Kill existing processes
    kill_existing_streamlit()
    
    # Open browser
    print("\n🌐 Opening browser to: http://localhost:8501")
    webbrowser.open("http://localhost:8501")
    
    print("\n✅ Starting Streamlit dashboard...")
    print("   Press Ctrl+C to stop when done")
    print("-"*60)
    
    try:
        # Start Streamlit
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py",
             "--server.port=8501", "--server.address=localhost"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        # Wait for process
        process.wait()
        return True
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_dashboard()