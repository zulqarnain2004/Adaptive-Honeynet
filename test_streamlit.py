#!/usr/bin/env python3
"""Test Streamlit installation and dashboard"""

import subprocess
import sys
import time
import webbrowser
import os

def test_streamlit():
    print("Testing Streamlit installation...")
    
    # Test import
    try:
        import streamlit as st
        print(f"✅ Streamlit version: {st.__version__}")
    except ImportError:
        print("❌ Streamlit not installed")
        print("Install with: pip install streamlit")
        return False
    
    # Create a minimal test app
    test_app = """
import streamlit as st
import time

st.set_page_config(page_title="Test", page_icon="✅")
st.title("✅ Streamlit Test")
st.success("Streamlit is working correctly!")
st.info("Dashboard will open at http://localhost:8501")

# Show system info
col1, col2 = st.columns(2)
with col1:
    st.metric("Status", "Running", "Online")
with col2:
    st.metric("Port", "8501", "0")

st.balloons()
"""
    
    with open("test_app.py", "w") as f:
        f.write(test_app)
    
    print("\nStarting test server...")
    print("Dashboard will open in your browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Start Streamlit
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "test_app.py", 
             "--server.port=8501", "--server.address=0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Try to open browser
        webbrowser.open("http://localhost:8501")
        
        print("Server started. Press Ctrl+C to stop...")
        
        # Keep running
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\nStopping server...")
            proc.terminate()
            proc.wait()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists("test_app.py"):
            os.remove("test_app.py")
    
    return True

if __name__ == "__main__":
    test_streamlit()