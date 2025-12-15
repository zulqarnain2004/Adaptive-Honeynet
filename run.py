#!/usr/bin/env python3
"""
Simple runner for Adaptive Deception Mesh
"""
import os
import sys
import time
import webbrowser
import threading

def start_everything():
    print("="*70)
    print("Adaptive Deception Mesh - All-in-One Starter")
    print("="*70)
    
    # Check requirements
    try:
        import flask
        import rich
        print("✓ Requirements check passed")
    except ImportError:
        print("✗ Missing requirements. Installing...")
        os.system("pip install -r requirements.txt")
    
    # Create directories
    directories = ['models/saved_models', 'data', 'logs', 'mlflow_tracking', 'configs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Start server in background thread
    def start_server():
        from app import create_app
        app = create_app()
        
        ports = [5000, 5001, 5002]
        for port in ports:
            try:
                print(f"\n🌐 Starting server on port {port}...")
                app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
                break
            except Exception as e:
                if port == ports[-1]:
                    print(f"❌ All ports failed: {e}")
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("\n⏳ Waiting for server to start...")
    import requests
    
    for i in range(30):
        try:
            response = requests.get("http://localhost:5000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server is running!")
                
                # Open browser
                time.sleep(1)
                webbrowser.open("http://localhost:5000")
                
                # Start CLI
                print("\n" + "="*70)
                print("Starting CLI interface...")
                print("="*70)
                time.sleep(2)
                
                from cli import cli
                cli()
                break
        except:
            if i % 5 == 0:
                print(f"  Waiting... ({i+1}/30)")
            time.sleep(1)
    
    # Keep alive
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    start_everything()