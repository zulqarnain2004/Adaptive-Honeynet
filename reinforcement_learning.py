import numpy as np
import random
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class NetworkEnvironment(gym.Env):
    """Custom environment for honeynet defense"""
    
    def __init__(self, n_nodes: int = 10, max_honeypots: int = 3):
        super(NetworkEnvironment, self).__init__()
        
        self.n_nodes = n_nodes
        self.max_honeypots = max_honeypots
        
        # Action space: place/remove honeypot at each node + no action
        self.action_space = spaces.Discrete(n_nodes + 1)  # 0: no action, 1-n: toggle node i-1
        
        # Observation space: binary vectors for honeypots and attacks
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2 * n_nodes,), 
            dtype=np.float32
        )
        
        # Initialize network state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Randomly initialize honeypots (max 2)
        n_honeypots = random.randint(0, min(2, self.max_honeypots))
        self.honeypots = random.sample(range(self.n_nodes), n_honeypots)
        
        # Randomly initialize attacks (1-3 nodes under attack)
        n_attacks = random.randint(1, 3)
        self.attacks = random.sample(range(self.n_nodes), n_attacks)
        
        # Ensure honeypots and attacks don't overlap initially
        self.attacks = [a for a in self.attacks if a not in self.honeypots]
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        obs = np.zeros(2 * self.n_nodes, dtype=np.float32)
        
        # Set honeypot positions
        for hp in self.honeypots:
            obs[hp] = 1
        
        # Set attack positions
        for i, attack in enumerate(self.attacks):
            obs[self.n_nodes + attack] = 1
        
        return obs
    
    def step(self, action: int):
        """Execute one time step"""
        reward = 0
        done = False
        
        # Execute action
        if action > 0:  # Toggle honeypot at node (action-1)
            node = action - 1
            
            if node in self.honeypots:
                # Remove honeypot
                self.honeypots.remove(node)
                reward -= 0.2  # Penalty for removing honeypot unnecessarily
            else:
                # Add honeypot if we have capacity
                if len(self.honeypots) < self.max_honeypots:
                    self.honeypots.append(node)
                    reward -= 0.1  # Small cost for placing honeypot
                else:
                    reward -= 0.3  # Penalty for trying to exceed capacity
        
        # Simulate attack movement
        self._move_attacks()
        
        # Calculate rewards
        detected_attacks = set(self.attacks) & set(self.honeypots)
        
        if detected_attacks:
            # Positive reward for detecting attacks
            reward += len(detected_attacks) * 1.0
            # Remove detected attacks
            self.attacks = [a for a in self.attacks if a not in detected_attacks]
        
        # Penalty for false positives (honeypots without attacks)
        false_positives = len([hp for hp in self.honeypots if hp not in self.attacks])
        reward -= false_positives * 0.5
        
        # Penalty for missed attacks
        missed_attacks = len([a for a in self.attacks if a not in self.honeypots])
        reward -= missed_attacks * 0.3
        
        # Game ends if all attacks are detected or too many steps
        if len(self.attacks) == 0:
            reward += 5  # Bonus for clearing all attacks
            done = True
        
        # Maximum steps check
        if self.current_step >= 100:
            done = True
        
        self.current_step += 1
        
        return self._get_observation(), reward, done, {}
    
    def _move_attacks(self):
        """Simulate attacker moving to adjacent nodes"""
        new_attacks = []
        
        for attack in self.attacks:
            # Attackers might move to adjacent nodes (simplified)
            if random.random() < 0.3:  # 30% chance to move
                # Move to random adjacent node (simulating network topology)
                possible_moves = [(attack + i) % self.n_nodes for i in [-1, 1, 0]]
                new_pos = random.choice(possible_moves)
                
                # Don't move to a node with honeypot
                if new_pos not in self.honeypots:
                    new_attacks.append(new_pos)
                else:
                    new_attacks.append(attack)  # Stay if moving to honeypot
            else:
                new_attacks.append(attack)
        
        self.attacks = list(set(new_attacks))  # Remove duplicates
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step: {self.current_step}")
        print(f"Honeypots: {sorted(self.honeypots)}")
        print(f"Attacks: {sorted(self.attacks)}")
        print(f"Detected: {sorted(set(self.attacks) & set(self.honeypots))}")
        print("-" * 30)

class QLearningAgent:
    """Q-Learning agent for honeypot placement"""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.1, epsilon_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def state_to_key(self, state):
        """Convert state to hashable key"""
        return tuple(state.astype(int))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_key = self.state_to_key(state)
        
        if random.random() < self.epsilon:
            # Exploration
            return self.env.action_space.sample()
        else:
            # Exploitation
            return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Next max Q-value
        next_max_q = np.max(self.q_table[next_state_key]) if not done else 0
        
        # Update Q-value using Bellman equation
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes=1000, render_every=100):
        """Train the agent"""
        rewards_history = []
        epsilon_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            rewards_history.append(total_reward)
            epsilon_history.append(self.epsilon)
            
            if (episode + 1) % render_every == 0:
                avg_reward = np.mean(rewards_history[-render_every:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return rewards_history, epsilon_history
    
    def save_q_table(self, filename="models/q_table.npy"):
        """Save Q-table"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self, filename="models/q_table.npy"):
        """Load Q-table"""
        import pickle
        with open(filename, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), q_table_dict)

class DQNAgent(nn.Module):
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNAgent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def act(self, state, epsilon=0.1):
        """Choose action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self(state_tensor)
            return torch.argmax(q_values).item()

def train_dqn_agent(env, episodes=500, batch_size=32, gamma=0.99,
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """Train DQN agent"""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    target_agent = DQNAgent(state_size, action_size)
    target_agent.load_state_dict(agent.state_dict())
    
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    replay_buffer = []
    replay_capacity = 10000
    
    epsilon = epsilon_start
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.act(state, epsilon)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > replay_capacity:
                replay_buffer.pop(0)
            
            # Train
            if len(replay_buffer) >= batch_size:
                # Sample batch
                batch = random.sample(replay_buffer, batch_size)
                
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)
                
                # Current Q values
                current_q = agent(states_tensor).gather(1, actions_tensor)
                
                # Next Q values
                with torch.no_grad():
                    next_q = target_agent(next_states_tensor).max(1)[0]
                    target_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)
                
                # Compute loss
                loss = criterion(current_q.squeeze(), target_q)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            total_reward += reward
        
        # Update target network
        if episode % 10 == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}")
    
    return agent, rewards_history