#!/usr/bin/env python3
"""
Setup script for Adaptive Deception-Mesh
This helps users get started quickly.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🛡️  ADAPTIVE DECEPTION-MESH SETUP")
    print("=" * 70)
    print("An Intelligent Honeynet with Learning and Adaptation")
    print("CS351 - Artificial Intelligence Project")
    print("GIK Institute of Engineering Sciences and Technology")
    print("=" * 70)

def check_python_version():
    """Check Python version"""
    print("\n🔍 Checking Python version...")
    
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python 3.8 or higher required")
        return False
    
    print("  ✅ Python version OK")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # List of core dependencies
    core_deps = [
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "streamlit==1.28.0",
        "networkx==3.1",
        "gymnasium==0.29.1",
        "pyyaml==6.0",
        "scipy==1.11.3",
        "joblib==1.3.2",
        "imbalanced-learn==0.11.0"
    ]
    
    # Optional dependencies (for advanced features)
    optional_deps = [
        "shap==0.42.1",
        "lime==0.2.0.1",
        "plotly==5.17.0",
        "mlflow==2.7.1"
    ]
    
    print("  Installing core dependencies...")
    try:
        for dep in core_deps:
            print(f"    Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        print("\n  Installing optional dependencies...")
        for dep in optional_deps:
            print(f"    Installing {dep}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            except:
                print(f"    ⚠️  Could not install {dep} (optional)")
        
        print("\n✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install dependencies: {e}")
        return False

def create_project_structure():
    """Create project folder structure"""
    print("\n🏗️  Creating project structure...")
    
    # List of directories to create
    directories = [
        'data/raw',
        'data/processed',
        'data/logs',
        'models/saved_models',
        'models/results',
        'src',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📁 Created: {directory}/")
    
    # Create __init__.py files
    init_files = ['src/__init__.py', 'tests/__init__.py']
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")
        print(f"  📄 Created: {init_file}")
    
    return True

def create_config_file():
    """Create configuration file"""
    print("\n⚙️  Creating configuration file...")
    
    config_content = """# Adaptive Deception-Mesh Configuration

data:
  raw_path: "data/raw/UNSW_NB15_training-set.csv"
  processed_path: "data/processed/"
  test_size: 0.3
  random_state: 42

model:
  ml_models:
    - random_forest
    - logistic_regression
  clustering:
    n_clusters: 5
  reinforcement:
    learning_rate: 0.1
    discount_factor: 0.9
    epsilon: 0.1
    episodes: 500

network:
  nodes: 10
  max_honeypots: 3
  resources:
    cpu_min: 1
    cpu_max: 4
    ram_min: 2
    ram_max: 16
    bandwidth_min: 100
    bandwidth_max: 1000

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  k_folds: 3

streamlit:
  port: 8501
  theme: "light"
"""
    
    with open('config.yaml', 'w') as f:
        f.write(config_content)
    
    print("  📄 Created: config.yaml")
    return True

def create_sample_dataset():
    """Create a sample dataset if real one is not available"""
    print("\n📊 Checking dataset...")
    
    dataset_path = 'data/raw/UNSW_NB15_training-set.csv'
    
    if os.path.exists(dataset_path):
        print(f"  ✅ Dataset found: {dataset_path}")
        return True
    
    print("  ⚠️  No dataset found. Creating sample dataset...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create realistic sample data
        n_samples = 1000
        
        # Define attack types (for realism)
        attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        
        data = {
            'dur': np.random.exponential(0.5, n_samples),
            'proto': np.random.choice(['tcp', 'udp', 'icmp', 'arp'], n_samples, p=[0.7, 0.2, 0.05, 0.05]),
            'service': np.random.choice(['http', 'ftp', 'ssh', 'smtp', 'dns', '-'], n_samples),
            'state': np.random.choice(['FIN', 'INT', 'CON', 'REQ', 'RST'], n_samples),
            'spkts': np.random.randint(1, 200, n_samples),
            'dpkts': np.random.randint(0, 150, n_samples),
            'sbytes': np.random.randint(100, 20000, n_samples),
            'dbytes': np.random.randint(0, 15000, n_samples),
            'rate': np.random.uniform(0.1, 1000, n_samples),
            'sttl': np.random.randint(32, 255, n_samples),
            'dttl': np.random.randint(0, 255, n_samples),
            'sload': np.random.uniform(1000, 1000000, n_samples),
            'dload': np.random.uniform(0, 500000, n_samples),
            'sloss': np.random.randint(0, 10, n_samples),
            'dloss': np.random.randint(0, 10, n_samples),
            'sinpkt': np.random.uniform(0.001, 10, n_samples),
            'dinpkt': np.random.uniform(0.001, 10, n_samples),
            'sjit': np.random.uniform(0, 100, n_samples),
            'djit': np.random.uniform(0, 100, n_samples),
            'label': np.random.randint(0, 2, n_samples),
            'attack_cat': np.random.choice(attack_types, n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        }
        
        df = pd.DataFrame(data)
        df.to_csv(dataset_path, index=False)
        
        print(f"  ✅ Created sample dataset with {n_samples} records")
        print(f"     Attack distribution: {dict(df['attack_cat'].value_counts())}")
        
        return True
        
    except ImportError:
        print("  ❌ Could not create sample dataset. Install pandas and numpy first:")
        print("     pip install pandas numpy")
        return False
    except Exception as e:
        print(f"  ❌ Error creating dataset: {e}")
        return False

def create_readme():
    """Create README file"""
    print("\n📖 Creating documentation...")
    
    readme_content = """# 🛡️ Adaptive Deception-Mesh

An Intelligent Honeynet with Learning and Adaptation for Cybersecurity Defense.

## 🎯 Project Overview

This project implements an adaptive honeynet system that:
- Automatically detects cyber attacks using Machine Learning
- Intelligently places honeypots using Search Algorithms
- Optimizes resources using Constraint Satisfaction Problems
- Adapts defense using Reinforcement Learning
- Explains decisions using Explainable AI techniques

## 🚀 Quick Start

### 1. Setup
```bash
python setup.py
# OR
python run.py --mode setup