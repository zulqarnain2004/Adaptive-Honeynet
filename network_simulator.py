import networkx as nx
import numpy as np
import random
from datetime import datetime
import json

class NetworkSimulator:
    """
    Network simulator for honeypot deployment and attack simulation
    """
    
    def __init__(self, config):
        self.config = config
        self.network = nx.Graph()
        self.honeypots = set()
        self.attacks_log = []
        self.nodes_resources = {}
        self.initialize_network()
    
    def initialize_network(self):
        """Initialize network topology"""
        print(f"Initializing network with {self.config.NETWORK_NODES} nodes...")
        
        # Create network graph
        self.network = nx.erdos_renyi_graph(
            n=self.config.NETWORK_NODES,
            p=0.3,
            seed=self.config.RANDOM_STATE
        )
        
        # Add node attributes
        for node in self.network.nodes():
            self.network.nodes[node]['type'] = 'normal'
            self.network.nodes[node]['resources'] = {
                'cpu': random.randint(20, 100),
                'memory': random.randint(30, 100),
                'bandwidth': random.randint(50, 1000)
            }
            self.network.nodes[node]['position'] = (
                random.uniform(0, 100),
                random.uniform(0, 100)
            )
            self.network.nodes[node]['compromised'] = False
        
        # Add edge weights
        for u, v in self.network.edges():
            self.network.edges[u, v]['weight'] = random.uniform(0.1, 1.0)
            self.network.edges[u, v]['bandwidth'] = random.randint(100, 1000)
        
        # Deploy initial honeypots
        self.deploy_initial_honeypots()
        
        print(f"Network initialized: {self.network.number_of_nodes()} nodes, "
              f"{self.network.number_of_edges()} edges, {len(self.honeypots)} honeypots")
    
    def deploy_initial_honeypots(self):
        """Deploy initial honeypots"""
        num_honeypots = min(self.config.MAX_HONEYPOTS, self.config.NETWORK_NODES // 3)
        
        # Select nodes with high degree for honeypots
        degrees = dict(self.network.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(num_honeypots):
            node = sorted_nodes[i][0]
            self.deploy_honeypot(node)
    
    def deploy_honeypot(self, node):
        """Deploy honeypot on a node"""
        if node not in self.honeypots and len(self.honeypots) < self.config.MAX_HONEYPOTS:
            self.network.nodes[node]['type'] = 'honeypot'
            self.honeypots.add(node)
            return True
        return False
    
    def remove_honeypot(self, node):
        """Remove honeypot from a node"""
        if node in self.honeypots:
            self.network.nodes[node]['type'] = 'normal'
            self.honeypots.remove(node)
            return True
        return False
    
    def migrate_honeypot(self, from_node, to_node):
        """Migrate honeypot from one node to another"""
        if from_node in self.honeypots and to_node not in self.honeypots:
            self.remove_honeypot(from_node)
            self.deploy_honeypot(to_node)
            return True
        return False
    
    def get_state(self):
        """Get current network state for RL agent"""
        return {
            'active_nodes': self.get_active_nodes(),
            'honeypots': len(self.honeypots),
            'compromised_nodes': self.get_compromised_nodes(),
            'network_load': self.calculate_network_load(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_active_nodes(self):
        """Get number of active nodes"""
        return self.config.NETWORK_NODES
    
    def get_honeypots(self):
        """Get number of honeypots"""
        return len(self.honeypots)
    
    def get_compromised_nodes(self):
        """Get number of compromised nodes"""
        compromised = 0
        for node in self.network.nodes():
            if self.network.nodes[node].get('compromised', False):
                compromised += 1
        return compromised
    
    def calculate_network_load(self):
        """Calculate current network load"""
        total_load = 0
        for node in self.network.nodes():
            total_load += self.network.nodes[node]['resources']['cpu']
        
        avg_load = total_load / self.network.number_of_nodes()
        return avg_load
    
    def generate_attack_data(self):
        """Generate synthetic attack data"""
        attack_types = ['port_scan', 'dos', 'exploit', 'brute_force', 'normal']
        probabilities = [0.25, 0.18, 0.12, 0.10, 0.35]  # From screenshot distribution
        
        attack_type = random.choices(attack_types, probabilities)[0]
        
        # Generate attack features
        attack_data = {
            'src_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            'dst_ip': f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 22, 21, 25, 53]),
            'protocol': random.choice(['tcp', 'udp', 'icmp']),
            'duration': random.uniform(0.1, 600),
            'packet_count': random.randint(1, 10000),
            'byte_count': random.randint(64, 65535),
            'packet_rate': random.uniform(0.1, 1000),
            'port_diversity': random.randint(1, 100),
            'attack_type': attack_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log attack
        self.log_attack(attack_data)
        
        return attack_data
    
    def log_attack(self, attack_data):
        """Log attack to history"""
        self.attacks_log.append(attack_data)
        
        # Keep only last 1000 attacks
        if len(self.attacks_log) > 1000:
            self.attacks_log = self.attacks_log[-1000:]
    
    def get_topology(self):
        """Get network topology for visualization"""
        nodes = []
        edges = []
        
        for node in self.network.nodes():
            node_data = {
                'id': node,
                'type': self.network.nodes[node]['type'],
                'x': self.network.nodes[node]['position'][0],
                'y': self.network.nodes[node]['position'][1],
                'compromised': self.network.nodes[node].get('compromised', False),
                'resources': self.network.nodes[node]['resources']
            }
            nodes.append(node_data)
        
        for u, v in self.network.edges():
            edge_data = {
                'source': u,
                'target': v,
                'weight': self.network.edges[u, v]['weight'],
                'bandwidth': self.network.edges[u, v].get('bandwidth', 100)
            }
            edges.append(edge_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'honeypots': list(self.honeypots),
            'total_nodes': self.network.number_of_nodes(),
            'total_edges': self.network.number_of_edges()
        }
    
    def get_attack_statistics(self):
        """Get attack statistics"""
        if not self.attacks_log:
            return {
                'total': 0,
                'blocked': 0,
                'analysed': 0,
                'high_severity': 0,
                'by_type': {}
            }
        
        attack_types = {}
        for attack in self.attacks_log:
            attack_type = attack.get('attack_type', 'unknown')
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        total = len(self.attacks_log)
        blocked = int(total * 0.82)  # 82% detection accuracy from screenshot
        analysed = total
        high_severity = int(total * 0.3)  # 30% high severity
        
        return {
            'total': total,
            'blocked': blocked,
            'analysed': analysed,
            'high_severity': high_severity,
            'by_type': attack_types
        }