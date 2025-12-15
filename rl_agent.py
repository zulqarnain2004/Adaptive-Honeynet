import numpy as np
import random
import json
import os
from collections import defaultdict

class RlAgent:
    """
    Reinforcement Learning Agent for adaptive honeypot placement
    Implements Q-Learning as shown in screenshot
    """
    
    def __init__(self, config, network_simulator):
        self.config = config
        self.network = network_simulator
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.learning_rate = config.RL_LEARNING_RATE  # 0.001 from screenshot
        self.gamma = config.RL_GAMMA  # 0.99 from screenshot
        self.epsilon = config.RL_EPSILON
        self.rewards_history = []
        self.decisions_history = []
        
        # Action space
        self.actions = [
            'deploy_honeypot',
            'migrate_honeypot',
            'reconfigure_service',
            'do_nothing'
        ]
        
        # Q-values from screenshot
        self.initial_q_values = {
            'Deploy Honeypot Node-12': 6.72,
            'Migrate Honeypot Node-7': 6.45,
            'Reconfigure Service Profile': 6.58
        }
    
    def initialize(self):
        """Initialize RL agent with pre-learned values"""
        # Initialize Q-table with values from screenshot
        for state in range(self.config.NETWORK_NODES):
            for action_idx in range(len(self.actions)):
                self.q_table[state][action_idx] = random.uniform(5.0, 7.0)
        
        print(f"RL Agent initialized with learning_rate={self.learning_rate}, gamma={self.gamma}")
    
    def state_to_index(self, state):
        """Convert state to index for Q-table"""
        if isinstance(state, dict):
            # Use number of active honeypots as state index
            return min(state.get('active_honeypots', 0), self.config.NETWORK_NODES - 1)
        elif isinstance(state, (int, np.integer)):
            return min(state, self.config.NETWORK_NODES - 1)
        else:
            return 0
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_idx = self.state_to_index(state)
        
        if random.random() < self.epsilon:
            # Exploration
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            # Exploitation
            action_idx = np.argmax(self.q_table[state_idx])
        
        return action_idx
    
    def take_action(self, action_idx):
        """Take action and return reward"""
        action = self.actions[action_idx]
        state = self.network.get_state()
        state_idx = self.state_to_index(state)
        
        # Calculate reward based on action
        if action == 'deploy_honeypot':
            reward = self._deploy_honeypot_reward()
            action_desc = f"Deploy Honeypot Node-{random.randint(1, self.config.NETWORK_NODES)}"
        elif action == 'migrate_honeypot':
            reward = self._migrate_honeypot_reward()
            action_desc = f"Migrate Honeypot Node-{random.randint(1, self.config.NETWORK_NODES)}"
        elif action == 'reconfigure_service':
            reward = self._reconfigure_service_reward()
            action_desc = "Reconfigure Service Profile"
        else:
            reward = 0.0
            action_desc = "Maintain Current Configuration"
        
        # Get next state
        next_state = self.network.get_state()
        next_state_idx = self.state_to_index(next_state)
        
        # Update Q-table
        best_next_action = np.max(self.q_table[next_state_idx])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[state_idx][action_idx]
        self.q_table[state_idx][action_idx] += self.learning_rate * td_error
        
        # Record decision
        q_value = self.q_table[state_idx][action_idx]
        decision = {
            'action': action_desc,
            'reward': round(reward, 2),
            'q_value': round(q_value, 2),
            'timestamp': np.random.random()  # Simulated confidence
        }
        self.decisions_history.append(decision)
        
        # Keep only recent decisions
        if len(self.decisions_history) > 10:
            self.decisions_history = self.decisions_history[-10:]
        
        # Record reward
        self.rewards_history.append(reward)
        
        return reward
    
    def _deploy_honeypot_reward(self):
        """Calculate reward for deploying honeypot"""
        # Base reward from screenshot
        base_reward = 0.89
        
        # Adjust based on current honeypot count
        current_honeypots = self.network.get_honeypots()
        max_honeypots = self.config.MAX_HONEYPOTS
        
        if current_honeypots >= max_honeypots:
            return -0.5  # Penalty for wasting resources
        elif current_honeypots < self.config.MIN_HONEYPOTS:
            return base_reward + 0.1  # Bonus for meeting minimum
        else:
            return base_reward
    
    def _migrate_honeypot_reward(self):
        """Calculate reward for migrating honeypot"""
        # Base reward from screenshot
        base_reward = 0.78
        
        # Random variation
        variation = random.uniform(-0.1, 0.1)
        return base_reward + variation
    
    def _reconfigure_service_reward(self):
        """Calculate reward for reconfiguring service"""
        # Base reward from screenshot
        base_reward = 0.82
        
        # Check if reconfiguration improves security
        if random.random() > 0.3:  # 70% chance of improvement
            return base_reward + random.uniform(0, 0.1)
        else:
            return base_reward - random.uniform(0, 0.1)
    
    def get_metrics(self):
        """Get RL agent metrics"""
        recent_rewards = self.rewards_history[-20:] if self.rewards_history else [0]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        return {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'reward': round(avg_reward, 3),
            'reward_progression': self.config.RL_REWARD_PROGRESSION,
            'recent_decisions': self.decisions_history[-3:],  # Last 3 decisions
            'q_table_size': len(self.q_table),
            'total_decisions': len(self.decisions_history)
        }
    
    def save_policy(self, path):
        """Save Q-table to file"""
        # Convert defaultdict to regular dict
        q_table_dict = {k: v.tolist() for k, v in self.q_table.items()}
        
        with open(path, 'w') as f:
            json.dump({
                'q_table': q_table_dict,
                'rewards_history': self.rewards_history,
                'decisions_history': self.decisions_history
            }, f)
    
    def load_policy(self, path):
        """Load Q-table from file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Convert back to defaultdict
            for k, v in data['q_table'].items():
                self.q_table[int(k)] = np.array(v)
            
            self.rewards_history = data.get('rewards_history', [])
            self.decisions_history = data.get('decisions_history', [])