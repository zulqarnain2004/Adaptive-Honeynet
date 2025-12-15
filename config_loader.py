"""
Configuration loader and manager - UPDATED VERSION
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import dotenv

class ConfigLoader:
    """
    Load and manage configuration files
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_environment()
        
        # Create default configs if directory doesn't exist
        if not self.config_dir.exists():
            self.create_default_configs()
        
    def _load_environment(self):
        """Load environment variables"""
        # Load .env file if exists
        dotenv.load_dotenv()
        
        # Set default environment variables
        os.environ.setdefault('FLASK_ENV', 'development')
        os.environ.setdefault('SECRET_KEY', 'adaptive-deception-mesh-secret')
        
    def load_all_configs(self):
        """Load all configuration files"""
        config_files = self.list_configs()
        
        for config_name in config_files:
            self.load_config(config_name)
        
        return self.configs
    
    def load_config(self, config_name: str, config_type: str = 'yaml') -> Dict:
        """
        Load configuration file
        """
        config_path = self.config_dir / f"{config_name}.{config_type}"
        
        if not config_path.exists():
            # Try with different extensions
            for ext in ['yaml', 'yml', 'json']:
                alt_path = self.config_dir / f"{config_name}.{ext}"
                if alt_path.exists():
                    config_path = alt_path
                    config_type = ext
                    break
            else:
                raise FileNotFoundError(f"Config file not found: {config_name}")
        
        if config_type in ['yaml', 'yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_type == 'json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config type: {config_type}")
        
        # Store in cache
        self.configs[config_name] = config
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides to config"""
        def _apply(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Convert key to environment variable format
                    env_key = f"{prefix}_{key}".upper().replace("-", "_").replace(".", "_")
                    
                    # Handle nested dictionaries
                    if isinstance(value, dict):
                        _apply(value, env_key)
                    else:
                        # Check for environment variable
                        if env_key in os.environ:
                            env_value = os.environ[env_key]
                            
                            # Convert to appropriate type
                            if isinstance(value, bool):
                                config[key] = env_value.lower() in ('true', '1', 'yes')
                            elif isinstance(value, int):
                                config[key] = int(env_value)
                            elif isinstance(value, float):
                                config[key] = float(env_value)
                            elif isinstance(value, list):
                                if env_value.startswith('[') and env_value.endswith(']'):
                                    # JSON array
                                    config[key] = json.loads(env_value)
                                else:
                                    # Comma-separated list
                                    config[key] = [item.strip() for item in env_value.split(',')]
                            else:
                                config[key] = env_value
            
            return config
        
        return _apply(config)
    
    def save_config(self, config_name: str, config: Dict, config_type: str = 'yaml'):
        """Save configuration to file"""
        config_path = self.config_dir / f"{config_name}.{config_type}"
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_type in ['yaml', 'yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif config_type == 'json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config type: {config_type}")
        
        # Update cache
        self.configs[config_name] = config
    
    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """Get a specific configuration value"""
        if config_name not in self.configs:
            self.load_config(config_name)
        
        config = self.configs[config_name]
        
        if key is None:
            return config
        
        # Navigate nested keys (e.g., "database.host")
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, config_name: str, key: str, value: Any):
        """Set a configuration value"""
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        config = self.configs[config_name]
        
        # Navigate/create nested structure
        keys = key.split('.')
        current = config
        
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set final value
        current[keys[-1]] = value
        
        # Auto-save
        self.save_config(config_name, config)
    
    def list_configs(self) -> list:
        """List all available configurations"""
        config_files = []
        
        if self.config_dir.exists():
            for ext in ['yaml', 'yml', 'json']:
                config_files.extend(self.config_dir.glob(f'*.{ext}'))
        
        return sorted([f.stem for f in config_files])
    
    def create_default_configs(self):
        """Create default configuration files"""
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        print(f"Creating default configurations in {self.config_dir}")
        
        # Return success - actual configs are created separately
        return True
    
    def export_config(self, format: str = 'json') -> str:
        """Export all configurations in specified format"""
        all_configs = {}
        
        for config_name in self.list_configs():
            all_configs[config_name] = self.get(config_name)
        
        if format == 'json':
            return json.dumps(all_configs, indent=2)
        elif format == 'yaml':
            return yaml.dump(all_configs, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def validate_config(self, config_name: str, schema: Dict = None) -> Dict:
        """Validate configuration against schema"""
        config = self.get(config_name)
        
        if schema is None:
            # Simple validation - just check required fields based on config type
            if config_name == 'app':
                required = ['debug', 'host', 'port']
            elif config_name == 'database':
                required = ['type', 'path']
            elif config_name == 'ml':
                required = ['training', 'models']
            else:
                required = []
        else:
            required = schema.get('required', [])
        
        errors = []
        warnings = []
        
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config_name': config_name
        }