"""
Data validation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class DataValidator:
    """
    Validate datasets for quality and consistency
    """
    
    def __init__(self):
        self.validation_rules = {
            'unswnb15': self._validate_unswnb15,
            'cicids2017': self._validate_cicids2017,
            'cowrie': self._validate_cowrie_logs
        }
    
    def validate_dataset(self, df: pd.DataFrame, dataset_type: str) -> Dict:
        """
        Validate dataset based on type
        """
        if dataset_type not in self.validation_rules:
            return {
                'valid': False,
                'errors': [f'Unknown dataset type: {dataset_type}'],
                'warnings': []
            }
        
        return self.validation_rules[dataset_type](df)
    
    def _validate_unswnb15(self, df: pd.DataFrame) -> Dict:
        """
        Validate UNSW-NB15 dataset
        """
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        numeric_columns = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                warnings.append(f"Column {col} is not numeric")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            warnings.append(f"Dataset contains {missing_values} missing values")
        
        # Check label distribution if label column exists
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            if len(label_counts) < 2:
                warnings.append("Dataset has only one class label")
            
            # Check for extreme imbalance
            if len(label_counts) == 2:
                ratio = label_counts.max() / label_counts.min()
                if ratio > 10:
                    warnings.append(f"Severe class imbalance: {ratio:.1f}:1 ratio")
        
        # Check attack category distribution
        if 'attack_cat' in df.columns:
            attack_counts = df['attack_cat'].value_counts()
            if 'Normal' in attack_counts:
                normal_pct = attack_counts['Normal'] / len(df) * 100
                if normal_pct < 30 or normal_pct > 70:
                    warnings.append(f"Normal traffic percentage unusual: {normal_pct:.1f}%")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': missing_values,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
    
    def _validate_cicids2017(self, df: pd.DataFrame) -> Dict:
        """
        Validate CICIDS2017 dataset
        """
        errors = []
        warnings = []
        
        # Basic checks
        if len(df) == 0:
            errors.append("Dataset is empty")
        
        # Check for required flow features
        flow_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
        missing_flow = [feat for feat in flow_features if feat not in df.columns]
        
        if missing_flow:
            warnings.append(f"Missing flow features: {missing_flow}")
        
        # Check protocol distribution
        if 'Protocol' in df.columns:
            protocol_counts = df['Protocol'].value_counts()
            if len(protocol_counts) < 2:
                warnings.append("Limited protocol diversity")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'rows': len(df),
                'columns': len(df.columns)
            }
        }
    
    def _validate_cowrie_logs(self, df: pd.DataFrame) -> Dict:
        """
        Validate Cowrie honeypot logs
        """
        errors = []
        warnings = []
        
        required_columns = ['timestamp', 'src_ip', 'eventid']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check timestamp format
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                warnings.append("Timestamp column may have invalid formats")
        
        # Check event distribution
        if 'eventid' in df.columns:
            event_counts = df['eventid'].value_counts()
            if len(event_counts) < 2:
                warnings.append("Limited event type diversity")
            
            # Check for suspicious activity patterns
            suspicious_events = ['cowrie.login.success', 'cowrie.command.input']
            suspicious_count = sum(df['eventid'].isin(suspicious_events))
            if suspicious_count / len(df) > 0.8:
                warnings.append("High proportion of suspicious events")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'rows': len(df),
                'columns': len(df.columns)
            }
        }
    
    def detect_anomalies(self, df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """
        Detect anomalies in numeric columns
        """
        anomalies = pd.DataFrame()
        
        for col in numeric_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Calculate statistics
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Define bounds
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Find anomalies
                col_anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not col_anomalies.empty:
                    col_anomalies = col_anomalies.copy()
                    col_anomalies['anomaly_column'] = col
                    col_anomalies['anomaly_value'] = col_anomalies[col]
                    col_anomalies['expected_range'] = f"({lower_bound:.2f}, {upper_bound:.2f})"
                    
                    anomalies = pd.concat([anomalies, col_anomalies])
        
        return anomalies
    
    def validate_feature_ranges(self, df: pd.DataFrame, feature_rules: Dict) -> Dict:
        """
        Validate feature value ranges
        """
        violations = []
        
        for feature, rules in feature_rules.items():
            if feature in df.columns:
                if 'min' in rules and df[feature].min() < rules['min']:
                    violations.append(f"{feature}: min value {df[feature].min()} < {rules['min']}")
                
                if 'max' in rules and df[feature].max() > rules['max']:
                    violations.append(f"{feature}: max value {df[feature].max()} > {rules['max']}")
                
                if 'allowed_values' in rules:
                    invalid_values = set(df[feature].unique()) - set(rules['allowed_values'])
                    if invalid_values:
                        violations.append(f"{feature}: invalid values {invalid_values}")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }