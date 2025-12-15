#!/usr/bin/env python3
"""
Setup script for Adaptive Deception Mesh
"""

import os
import shutil
import sys

def setup_project():
    print("=" * 60)
    print("ADAPTIVE DECEPTION MESH - PROJECT SETUP")
    print("=" * 60)
    
    # Create directories
    directories = [
        'models/saved_models',
        'data',
        'logs',
        'mlflow_tracking',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Check for requirements
    print("\nChecking requirements...")
    
    try:
        import numpy
        import pandas
        import sklearn
        import flask
        import mlflow
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return 1
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Train models: python train_models.py")
    print("2. Start server: python cli.py start")
    print("3. View system: python cli.py all")
    
    return 0

if __name__ == '__main__':
    sys.exit(setup_project())