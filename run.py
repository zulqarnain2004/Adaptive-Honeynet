#!/usr/bin/env python3
"""
Adaptive Deception-Mesh Runner
Complete project execution script with enhanced UI
"""

import argparse
import sys
import os
import subprocess
import webbrowser
import time
from datetime import datetime
import json

def check_streamlit_version():
    """Check and display Streamlit version"""
    try:
        result = subprocess.run([sys.executable, "-m", "streamlit", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.strip()
            print(f"📦 Streamlit Version: {version_line}")
            
            # Extract version number
            import re
            version_match = re.search(r'(\d+\.\d+\.\d+)', version_line)
            if version_match:
                version = version_match.group(1)
                major, minor, patch = map(int, version.split('.'))
                
                if major >= 1 and minor >= 12:
                    print("✅ Streamlit version is compatible with icon parameter")
                else:
                    print("⚠️  Streamlit version may not support icon parameter")
                    print("   Consider upgrading: pip install --upgrade streamlit>=1.12.0")
            return True
        else:
            print("❌ Streamlit not found or error checking version")
            return False
    except Exception as e:
        print(f"❌ Error checking Streamlit version: {e}")
        return False

# ... [rest of the run.py code remains the same as previous version] ...

def run_dashboard():
    """Run the interactive dashboard"""
    print_step(1, "Starting Interactive Dashboard", "processing")
    
    # Check Streamlit version
    if not check_streamlit_version():
        print(f"   {Color.YELLOW}⚠{Color.ENDC} Streamlit version check failed")
        print(f"   {Color.YELLOW}   Installing latest Streamlit...{Color.ENDC}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "streamlit>=1.12.0"])
            print(f"   {Color.GREEN}✓{Color.ENDC} Streamlit updated successfully")
        except:
            print(f"   {Color.RED}✗{Color.ENDC} Failed to update Streamlit")
    
    # Display dashboard info
    dashboard_info = f"""
{Color.CYAN}┌────────────────────────────────────────────────────────────┐{Color.ENDC}
{Color.CYAN}│{Color.ENDC} {Color.BOLD}🌐 DASHBOARD INFORMATION:{Color.ENDC}                              {Color.CYAN}│{Color.ENDC}
{Color.CYAN}├────────────────────────────────────────────────────────────┤{Color.ENDC}
{Color.CYAN}│{Color.ENDC} {Color.BOLD}URL:{Color.ENDC} http://localhost:8501                           {Color.CYAN}│{Color.ENDC}
{Color.CYAN}│{Color.ENDC} {Color.BOLD}Status:{Color.ENDC} Starting...                                  {Color.CYAN}│{Color.ENDC}
{Color.CYAN}│{Color.ENDC} {Color.BOLD}Auto-open:{Color.ENDC} Enabled                                   {Color.CYAN}│{Color.ENDC}
{Color.CYAN}│{Color.ENDC} {Color.BOLD}Stop:{Color.ENDC} Press Ctrl+C                                   {Color.CYAN}│{Color.ENDC}
{Color.CYAN}└────────────────────────────────────────────────────────────┘{Color.ENDC}
    """
    print(dashboard_info)
    
    # Open browser
    try:
        print(f"   {Color.GREEN}✓{Color.ENDC} Opening browser...")
        webbrowser.open("http://localhost:8501")
        time.sleep(1)
    except:
        print(f"   {Color.YELLOW}⚠{Color.ENDC} Could not open browser automatically")
        print(f"   {Color.YELLOW}   Please open: http://localhost:8501 manually{Color.ENDC}")
    
    # Start Streamlit with simplified command
    print(f"\n{Color.CYAN}Starting Streamlit server...{Color.ENDC}")
    print(f"{Color.YELLOW}Dashboard is now running. Press Ctrl+C to stop.{Color.ENDC}")
    
    try:
        # Simplified command for compatibility
        cmd = [
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port=8501", "--server.address=0.0.0.0"
        ]
        
        # Try to add theme parameters if supported
        try:
            # Check if theme parameters are supported
            test_cmd = [sys.executable, "-m", "streamlit", "run", "--help"]
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            if "--theme.primaryColor" in result.stdout:
                cmd.extend(["--theme.primaryColor=#3949ab"])
                cmd.extend(["--theme.backgroundColor=#0f172a"])
                cmd.extend(["--theme.secondaryBackgroundColor=#1e293b"])
                cmd.extend(["--theme.textColor=#f8fafc"])
        except:
            pass  # Ignore if theme parameters not supported
        
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}Dashboard stopped by user{Color.ENDC}")
        return True
    except Exception as e:
        print(f"{Color.RED}✗ Dashboard error: {e}{Color.ENDC}")
        return False

# ... [rest of the file remains the same] ...