"""
Model management and versioning
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import mlflow
import mlflow.sklearn
from pathlib import Path

class ModelManager:
    """
    Manage ML model lifecycle
    """
    
    def __init__(self, config):
        self.config = config
        self.models_dir = config.MODEL_DIR
        self.metadata_file = os.path.join(self.models_dir, 'model_metadata.json')
        self.metadata = self.load_metadata()
        
        # MLflow setup
        mlflow.set_tracking_uri('file:./mlflow_tracking')
        
    def load_metadata(self) -> Dict:
        """Load model metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'versions': {},
            'active_models': {},
            'experiments': {}
        }
    
    def save_metadata(self):
        """Save model metadata"""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, model_name: str, model_type: str, 
                      model_path: str, metrics: Dict, 
                      features: List[str], version: str = None) -> str:
        """
        Register a new model version
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_id = f"{model_name}_{version}"
        
        model_info = {
            'name': model_name,
            'type': model_type,
            'path': model_path,
            'version': version,
            'metrics': metrics,
            'features': features,
            'created_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        # Update metadata
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {}
        
        self.metadata['models'][model_name][version] = model_info
        self.metadata['versions'][model_id] = model_info
        
        # Set as active if first version
        if model_name not in self.metadata['active_models']:
            self.metadata['active_models'][model_name] = version
        
        self.save_metadata()
        
        # Log to MLflow
        self._log_to_mlflow(model_name, model_type, metrics, version)
        
        return model_id
    
    def _log_to_mlflow(self, model_name: str, model_type: str, 
                      metrics: Dict, version: str):
        """Log model to MLflow"""
        try:
            mlflow.set_experiment(model_name)
            
            with mlflow.start_run(run_name=version):
                # Log parameters
                mlflow.log_param('model_type', model_type)
                mlflow.log_param('version', version)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                model_path = os.path.join(self.models_dir, f"{model_name}_{version}.joblib")
                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path)
                
                # Log metadata
                mlflow.log_dict(metrics, 'metrics.json')
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")
    
    def load_model(self, model_name: str, version: str = None) -> Any:
        """
        Load a model by name and version
        """
        if version is None:
            version = self.metadata['active_models'].get(model_name)
        
        if not version or model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.metadata['models'][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        model_info = self.metadata['models'][model_name][version]
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def get_active_model(self, model_name: str) -> Dict:
        """Get active model info"""
        if model_name not in self.metadata['active_models']:
            raise ValueError(f"No active model for {model_name}")
        
        version = self.metadata['active_models'][model_name]
        return self.metadata['models'][model_name][version]
    
    def set_active_model(self, model_name: str, version: str):
        """Set active model version"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.metadata['models'][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        self.metadata['active_models'][model_name] = version
        self.save_metadata()
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        models = []
        
        for model_name, versions in self.metadata['models'].items():
            active_version = self.metadata['active_models'].get(model_name)
            
            for version, info in versions.items():
                model_info = info.copy()
                model_info['is_active'] = (version == active_version)
                models.append(model_info)
        
        return models
    
    def delete_model(self, model_name: str, version: str):
        """Delete a model version"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.metadata['models'][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        # Remove model file
        model_info = self.metadata['models'][model_name][version]
        if os.path.exists(model_info['path']):
            os.remove(model_info['path'])
        
        # Update metadata
        del self.metadata['models'][model_name][version]
        
        # Remove from versions index
        model_id = f"{model_name}_{version}"
        if model_id in self.metadata['versions']:
            del self.metadata['versions'][model_id]
        
        # Update active model if this was active
        if self.metadata['active_models'].get(model_name) == version:
            if self.metadata['models'][model_name]:
                # Set another version as active
                new_version = list(self.metadata['models'][model_name].keys())[-1]
                self.metadata['active_models'][model_name] = new_version
            else:
                del self.metadata['active_models'][model_name]
        
        self.save_metadata()
    
    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")
        
        for version in [version1, version2]:
            if version not in self.metadata['models'][model_name]:
                raise ValueError(f"Version {version} not found for model {model_name}")
        
        info1 = self.metadata['models'][model_name][version1]
        info2 = self.metadata['models'][model_name][version2]
        
        comparison = {
            'model_name': model_name,
            'versions': {
                version1: info1,
                version2: info2
            },
            'metrics_comparison': {}
        }
        
        # Compare metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in info1['metrics'] and metric in info2['metrics']:
                val1 = info1['metrics'][metric]
                val2 = info2['metrics'][metric]
                diff = val2 - val1
                improvement = diff > 0
                
                comparison['metrics_comparison'][metric] = {
                    'version1': val1,
                    'version2': val2,
                    'difference': diff,
                    'improvement': improvement
                }
        
        return comparison
    
    def export_model(self, model_name: str, version: str, export_dir: str):
        """Export model with all dependencies"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.metadata['models'][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        model_info = self.metadata['models'][model_name][version]
        
        # Create export directory
        export_path = Path(export_dir) / f"{model_name}_v{version}"
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        import shutil
        shutil.copy2(model_info['path'], export_path / "model.joblib")
        
        # Create metadata file
        metadata = {
            'model_info': model_info,
            'export_date': datetime.now().isoformat(),
            'system_info': {
                'python_version': os.sys.version,
                'platform': os.sys.platform
            }
        }
        
        with open(export_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create requirements file
        requirements = [
            "scikit-learn>=1.0.0",
            "joblib>=1.0.0",
            "numpy>=1.20.0",
            "pandas>=1.3.0"
        ]
        
        with open(export_path / "requirements.txt", 'w') as f:
            f.write("\n".join(requirements))
        
        # Create example usage script
        example_script = f'''"""
Example usage for {model_name} v{version}
"""

import joblib
import numpy as np
import json

# Load model
model = joblib.load('model.joblib')

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model: {{metadata['model_info']['name']}}")
print(f"Version: {{metadata['model_info']['version']}}")
print(f"Metrics: {{metadata['model_info']['metrics']}}")

# Example prediction
# features should match: {{metadata['model_info']['features']}}
# example_features = np.array([[feature_values...]])
# prediction = model.predict(example_features)
'''

        with open(export_path / "example_usage.py", 'w') as f:
            f.write(example_script)
        
        return str(export_path)