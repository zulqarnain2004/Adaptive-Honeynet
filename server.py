#!/usr/bin/env python3
"""
Reliable Server Starter for Adaptive Deception Mesh
"""
import subprocess
import sys
import os
import time
import atexit
import signal
import threading

def kill_process(proc):
    """Kill a process"""
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=3)
    except:
        try:
            proc.kill()
        except:
            pass

def start_server_directly():
    """Start server directly (not as subprocess)"""
    print("\n" + "="*60)
    print("🚀 Starting Adaptive Deception Mesh Server")
    print("="*60)
    
    # Set environment variable to prevent reloader
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    
    # Import and run app
    from app import create_app
    
    app = create_app()
    
    # Try multiple ports
    ports = [5000, 5001, 5002, 5003]
    
    for port in ports:
        try:
            print(f"\nTrying port {port}...")
            app.run(
                host='0.0.0.0', 
                port=port, 
                debug=False, 
                use_reloader=False,
                threaded=True
            )
            break
        except Exception as e:
            print(f"Port {port} failed: {e}")
            if port == ports[-1]:
                print("\n❌ All ports failed. Exiting...")
                sys.exit(1)

if __name__ == "__main__":
    start_server_directly()