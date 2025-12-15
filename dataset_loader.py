import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
import requests
import zipfile
import io

class DatasetLoader:
    """
    Loader for cybersecurity datasets
    """
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
    
    def load_unswnb15(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset
        """
        if file_path is None:
            file_path = self.config.DATASET_PATH
        
        print(f"Loading UNSW-NB15 dataset from {file_path}...")
        
        if os.path.exists(file_path):
            try:
                # Try to load the dataset
                df = pd.read_csv(file_path)
                print(f"Successfully loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
                return df
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Creating synthetic dataset instead...")
                return self.create_synthetic_unswnb15()
        else:
            print(f"Dataset file not found: {file_path}")
            print("Creating synthetic dataset for demonstration...")
            return self.create_synthetic_unswnb15()
    
    def create_synthetic_unswnb15(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Create synthetic UNSW-NB15-like dataset for demonstration
        """
        print(f"Creating synthetic UNSW-NB15 dataset with {n_samples} samples...")
        
        np.random.seed(self.config.RANDOM_STATE)
        
        # Create synthetic features similar to UNSW-NB15
        data = {
            # Basic connection features
            'srcip': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'sport': np.random.randint(1024, 65535, n_samples),
            'dstip': [f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'dsport': np.random.choice([80, 443, 22, 21, 25, 53, 3389, 8080], n_samples),
            'proto': np.random.choice(['tcp', 'udp', 'icmp', 'arp'], n_samples, p=[0.6, 0.3, 0.08, 0.02]),
            'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'], n_samples),
            
            # Timing features
            'dur': np.random.exponential(10, n_samples),
            'stime': np.random.uniform(0, 86400, n_samples),
            'ltime': np.random.uniform(0, 86400, n_samples),
            
            # Byte features
            'sbytes': np.random.randint(100, 10000, n_samples),
            'dbytes': np.random.randint(100, 10000, n_samples),
            'sttl': np.random.randint(1, 255, n_samples),
            'dttl': np.random.randint(1, 255, n_samples),
            
            # Packet features
            'spkts': np.random.randint(1, 100, n_samples),
            'dpkts': np.random.randint(1, 100, n_samples),
            'sload': np.random.exponential(100, n_samples),
            'dload': np.random.exponential(100, n_samples),
            
            # TCP features
            'swin': np.random.randint(0, 65535, n_samples),
            'dwin': np.random.randint(0, 65535, n_samples),
            'stcpb': np.random.randint(0, 1000000, n_samples),
            'dtcpb': np.random.randint(0, 1000000, n_samples),
            
            # Statistical features
            'smeansz': np.random.randint(0, 1500, n_samples),
            'dmeansz': np.random.randint(0, 1500, n_samples),
            'sjit': np.random.exponential(10, n_samples),
            'djit': np.random.exponential(10, n_samples),
            'sintpkt': np.random.exponential(0.1, n_samples),
            'dintpkt': np.random.exponential(0.1, n_samples),
            
            # Connection features
            'tcprtt': np.random.exponential(0.1, n_samples),
            'synack': np.random.exponential(0.1, n_samples),
            'ackdat': np.random.exponential(0.1, n_samples),
            
            # Security features
            'is_sm_ips_ports': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'ct_state_ttl': np.random.randint(0, 10, n_samples),
            'ct_flw_http_mthd': np.random.randint(0, 10, n_samples),
            'is_ftp_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'ct_ftp_cmd': np.random.randint(0, 10, n_samples),
            'ct_srv_src': np.random.randint(0, 100, n_samples),
            'ct_srv_dst': np.random.randint(0, 100, n_samples),
            'ct_dst_ltm': np.random.randint(0, 100, n_samples),
            'ct_src_ltm': np.random.randint(0, 100, n_samples),
            'ct_src_dport_ltm': np.random.randint(0, 100, n_samples),
            'ct_dst_sport_ltm': np.random.randint(0, 100, n_samples),
            'ct_dst_src_ltm': np.random.randint(0, 100, n_samples),
            
            # Attack labels (matching screenshot distribution)
            'attack_cat': np.random.choice(['Normal', 'Exploits', 'DoS', 'PortScan'], n_samples, 
                                          p=[0.45, 0.12, 0.18, 0.25]),
            'label': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(data)
        print(f"Synthetic dataset created: {df.shape}")
        
        # Save for future use
        os.makedirs(os.path.dirname(self.config.DATASET_PATH), exist_ok=True)
        df.to_csv(self.config.DATASET_PATH, index=False)
        print(f"Dataset saved to {self.config.DATASET_PATH}")
        
        return df
    
    def load_cicids2017(self) -> pd.DataFrame:
        """
        Load CICIDS2017 dataset (simulated for demonstration)
        """
        print("Loading CICIDS2017 dataset...")
        # In a real implementation, you would load the actual dataset
        # For now, we'll create synthetic data
        return self.create_synthetic_unswnb15(n_samples=5000)
    
    def load_cowrie_logs(self) -> pd.DataFrame:
        """
        Load Cowrie honeypot logs (simulated for demonstration)
        """
        print("Loading Cowrie honeypot logs...")
        
        # Create synthetic honeypot logs
        n_samples = 2000
        np.random.seed(self.config.RANDOM_STATE)
        
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='T'),
            'src_ip': [f"203.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'dst_ip': '192.168.1.100',
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': np.random.choice([22, 23, 2222], n_samples),
            'username': np.random.choice(['root', 'admin', 'user', 'test', ''], n_samples),
            'password': np.random.choice(['password', '123456', 'admin', 'test', ''], n_samples),
            'command': np.random.choice(['ls', 'cd', 'whoami', 'cat', 'wget', 'curl', ''], n_samples),
            'input': [''] * n_samples,
            'eventid': np.random.choice(['cowrie.login.success', 'cowrie.login.failed', 
                                        'cowrie.command.input', 'cowrie.session.file_download'], n_samples),
            'duration': np.random.exponential(30, n_samples),
            'is_attack': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        return pd.DataFrame(data)
    
    def generate_synthetic_logs(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic logs for RL training
        """
        print(f"Generating {n_samples} synthetic logs for RL training...")
        
        np.random.seed(self.config.RANDOM_STATE)
        
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='S'),
            'node_id': np.random.randint(0, self.config.NETWORK_NODES, n_samples),
            'action_type': np.random.choice(['deploy', 'migrate', 'remove', 'reconfigure'], n_samples),
            'resource_usage_cpu': np.random.uniform(10, 90, n_samples),
            'resource_usage_memory': np.random.uniform(20, 80, n_samples),
            'attack_detected': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'false_positive': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'reward': np.random.uniform(-1, 1, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get information about a dataset
        """
        info = {
            'UNSW-NB15': {
                'name': 'UNSW-NB15',
                'description': 'Comprehensive network traffic dataset with various attack types',
                'samples': 10000,
                'features': 49,
                'attack_types': ['Normal', 'Exploits', 'DoS', 'PortScan'],
                'purpose': 'Train and test attack classification models'
            },
            'CICIDS2017': {
                'name': 'CICIDS2017',
                'description': 'Intrusion Detection Evaluation Dataset',
                'samples': 5000,
                'features': 78,
                'attack_types': ['BENIGN', 'DDoS', 'PortScan', 'Botnet'],
                'purpose': 'Validate detection performance'
            },
            'Cowrie': {
                'name': 'Cowrie Honeypot Logs',
                'description': 'SSH/Telnet honeypot logs',
                'samples': 2000,
                'features': 10,
                'attack_types': ['Brute Force', 'Command Injection', 'File Download'],
                'purpose': 'Model attacker behavior and simulate RL environment'
            }
        }
        
        return info.get(dataset_name, {})