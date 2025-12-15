import time
from datetime import datetime
import json
import os

class SystemMetrics:
    """
    System metrics tracker
    """
    
    def __init__(self, config):
        self.config = config
        self.metrics = {
            'start_time': time.time(),
            'total_attacks': 0,
            'blocked_attacks': 0,
            'analysed_attacks': 0,
            'high_severity': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'rl_total_reward': 0,
            'rl_episodes': 0,
            'cpu_usage_history': [],
            'memory_usage_history': [],
            'detection_accuracy_history': []
        }
        
        # From screenshot
        self.total_attacks = 0
        self.blocked_attacks = 0
        self.analysed_attacks = 0
        self.high_severity = 0
        
        self.simulation_start_time = None
        self.last_update = time.time()
    
    def update(self, attack_detected=False, rl_reward=0, 
               active_nodes=20, honeypots=8):
        """
        Update system metrics
        """
        current_time = time.time()
        
        # Update attack metrics (simulating from screenshot)
        if attack_detected:
            self.total_attacks += 1
            self.blocked_attacks += 1  # Assuming all detected attacks are blocked
            self.analysed_attacks += 1
        
        # Randomly add high severity attacks (30% chance)
        if attack_detected and time.time() % 10 < 3:
            self.high_severity += 1
        
        # Update RL metrics
        self.metrics['rl_total_reward'] += rl_reward
        self.metrics['rl_episodes'] += 1
        
        # Record resource usage (from screenshot values)
        self.metrics['cpu_usage_history'].append({
            'timestamp': current_time,
            'value': 45  # From screenshot
        })
        
        self.metrics['memory_usage_history'].append({
            'timestamp': current_time,
            'value': 52  # From screenshot
        })
        
        self.metrics['detection_accuracy_history'].append({
            'timestamp': current_time,
            'value': self.config.DETECTION_ACCURACY  # 82% from screenshot
        })
        
        # Keep only last 1000 entries
        for key in ['cpu_usage_history', 'memory_usage_history', 'detection_accuracy_history']:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
        
        self.last_update = current_time
    
    def log_simulation_step(self):
        """Log simulation step metrics"""
        step_metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_attacks': self.total_attacks,
            'blocked_attacks': self.blocked_attacks,
            'analysed_attacks': self.analysed_attacks,
            'high_severity': self.high_severity,
            'rl_avg_reward': self.get_rl_average_reward(),
            'system_uptime': self.get_uptime()
        }
        
        # Log to file
        self._log_to_file(step_metrics)
    
    def get_rl_average_reward(self):
        """Get average RL reward"""
        if self.metrics['rl_episodes'] == 0:
            return 0
        return self.metrics['rl_total_reward'] / self.metrics['rl_episodes']
    
    def get_uptime(self):
        """Get system uptime in seconds"""
        return time.time() - self.metrics['start_time']
    
    def get_summary(self):
        """Get metrics summary"""
        return {
            'system': {
                'uptime_seconds': self.get_uptime(),
                'total_attacks': self.total_attacks,
                'blocked_attacks': self.blocked_attacks,
                'block_rate': self.blocked_attacks / max(self.total_attacks, 1),
                'analysed_attacks': self.analysed_attacks,
                'high_severity': self.high_severity,
                'detection_accuracy': self.config.DETECTION_ACCURACY
            },
            'rl_agent': {
                'total_episodes': self.metrics['rl_episodes'],
                'total_reward': self.metrics['rl_total_reward'],
                'average_reward': self.get_rl_average_reward(),
                'current_reward': self.config.RL_AGENT_REWARD  # From screenshot
            },
            'resources': {
                'cpu_usage': 45,  # From screenshot
                'memory_usage': 52,  # From screenshot
                'active_nodes': 20,  # From screenshot
                'honeypots': 8  # From screenshot
            }
        }
    
    def _log_to_file(self, metrics):
        """Log metrics to file"""
        log_dir = 'logs/metrics'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'metrics.jsonl')
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def reset(self):
        """Reset all metrics"""
        self.__init__(self.config)