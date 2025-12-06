#!/usr/bin/env python3
"""
Simple demo runner for Adaptive Deception-Mesh
"""

import subprocess
import sys
import webbrowser
import time

def main():
    """Run the demo"""
    print("=" * 60)
    print("🛡️  ADAPTIVE DECEPTION-MESH - QUICK DEMO")
    print("=" * 60)
    print("\nStarting demo dashboard...")
    print("Dashboard will open in your browser automatically.")
    print("If it doesn't open, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the demo")
    print("=" * 60)
    
    # Open browser
    webbrowser.open("http://localhost:8501")
    
    # Start Streamlit
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py",
             "--server.port=8501", "--server.address=localhost",
             "--theme.base=light"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        process.wait()
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()