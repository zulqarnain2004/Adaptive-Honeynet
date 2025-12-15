import logging
import os
from datetime import datetime
import json

def setup_logger(app):
    """
    Setup application logger
    """
    log_dir = app.config['LOG_DIR']
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'app_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    app.logger = logging.getLogger('adaptive_deception_mesh')
    app.logger.info('Logger initialized')

class JSONLogger:
    """
    JSON logger for structured logging
    """
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'events_{timestamp}.jsonl')
    
    def log_event(self, event_type, data, level='INFO'):
        """
        Log event in JSON Lines format
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'level': level,
            'data': data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Also print to console for development
        print(f"[{level}] {event_type}: {json.dumps(data, indent=2)}")
    
    def log_attack(self, attack_data, prediction, confidence):
        """Log attack detection event"""
        self.log_event('attack_detection', {
            'attack': attack_data,
            'prediction': prediction,
            'confidence': confidence,
            'action_taken': 'logged'
        })
    
    def log_rl_action(self, action, reward, q_value):
        """Log RL agent action"""
        self.log_event('rl_action', {
            'action': action,
            'reward': reward,
            'q_value': q_value
        })
    
    def log_system_metrics(self, metrics):
        """Log system metrics"""
        self.log_event('system_metrics', metrics)