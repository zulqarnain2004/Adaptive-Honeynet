import networkx as nx
import heapq
from typing import List, Dict, Tuple, Set
import random

class NetworkNode:
    def __init__(self, node_id: int, resources: Dict):
        self.node_id = node_id
        self.resources = resources  # {'cpu': 4, 'ram': 16, 'bandwidth': 1000}
        self.honeypot = False
        self.attacker_present = False
        
    def can_host_honeypot(self, requirements: Dict) -> bool:
        """Check if node has sufficient resources for honeypot"""
        return all(self.resources[key] >= requirements[key] for key in requirements)
    
    def allocate_resources(self, requirements: Dict):
        """Allocate resources for honeypot"""
        for key in requirements:
            self.resources[key] -= requirements[key]
        self.honeypot = True
    
    def deallocate_resources(self, requirements: Dict):
        """Deallocate resources"""
        for key in requirements:
            self.resources[key] += requirements[key]
        self.honeypot = False

class AStarHoneypotPlanner:
    def __init__(self, network_graph: nx.Graph):
        self.graph = network_graph
        self.honeypot_requirements = {'cpu': 2, 'ram': 8, 'bandwidth': 500}
        
    def heuristic(self, node1: int, node2: int) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        pos1 = self.graph.nodes[node1].get('pos', (0, 0))
        pos2 = self.graph.nodes[node2].get('pos', (0, 0))
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def find_optimal_placement(self, start_node: int, target_nodes: List[int]) -> List[int]:
        """Find optimal path for honeypot placement using A*"""
        paths = []
        
        for target in target_nodes:
            open_set = []
            heapq.heappush(open_set, (0, start_node))
            
            g_score = {node: float('inf') for node in self.graph.nodes()}
            g_score[start_node] = 0
            
            f_score = {node: float('inf') for node in self.graph.nodes()}
            f_score[start_node] = self.heuristic(start_node, target)
            
            came_from = {}
            
            while open_set:
                _, current = heapq.heappop(open_set)
                
                if current == target:
                    # Reconstruct path
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start_node)
                    paths.append(path[::-1])
                    break
                
                for neighbor in self.graph.neighbors(current):
                    tentative_g_score = g_score[current] + 1  # Each edge has weight 1
                    
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, target)
                        
                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return paths
    
    def plan_honeypot_deployment(self, suspicious_nodes: List[int], max_honeypots: int = 3) -> List[int]:
        """Plan honeypot deployment using A* with resource constraints"""
        candidate_nodes = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if (node not in suspicious_nodes and 
                node_data.get('resources', {}).get('cpu', 0) >= self.honeypot_requirements['cpu'] and
                node_data.get('resources', {}).get('ram', 0) >= self.honeypot_requirements['ram']):
                
                # Calculate score based on proximity to suspicious nodes and available resources
                total_distance = sum(self.heuristic(node, suspicious) for suspicious in suspicious_nodes)
                resource_score = sum(node_data['resources'].values())
                
                score = resource_score / (total_distance + 1)  # Avoid division by zero
                candidate_nodes.append((score, node))
        
        # Select top nodes
        candidate_nodes.sort(reverse=True)
        selected_nodes = [node for _, node in candidate_nodes[:max_honeypots]]
        
        return selected_nodes

class HoneypotCSP:
    def __init__(self, network_nodes: List[NetworkNode], max_honeypots: int):
        self.nodes = network_nodes
        self.max_honeypots = max_honeypots
        self.honeypot_requirements = {'cpu': 2, 'ram': 8, 'bandwidth': 500}
        
    def is_valid_assignment(self, assignment: Dict[int, bool]) -> bool:
        """Check if assignment satisfies all constraints"""
        honeypot_count = sum(assignment.values())
        
        if honeypot_count > self.max_honeypots:
            return False
        
        # Check resource constraints
        for node_id, has_honeypot in assignment.items():
            if has_honeypot:
                node = self.nodes[node_id]
                if not node.can_host_honeypot(self.honeypot_requirements):
                    return False
        
        return True
    
    def solve_backtracking(self) -> Dict[int, bool]:
        """Solve CSP using backtracking"""
        assignment = {i: False for i in range(len(self.nodes))}
        
        def backtrack(current_node: int, current_assignment: Dict, honeypot_count: int):
            if current_node == len(self.nodes):
                return current_assignment if self.is_valid_assignment(current_assignment) else None
            
            if honeypot_count >= self.max_honeypots:
                # Try without honeypot
                current_assignment[current_node] = False
                return backtrack(current_node + 1, current_assignment.copy(), honeypot_count)
            
            # Try with honeypot
            current_assignment[current_node] = True
            if self.is_valid_assignment(current_assignment):
                result = backtrack(current_node + 1, current_assignment.copy(), honeypot_count + 1)
                if result:
                    return result
            
            # Try without honeypot
            current_assignment[current_node] = False
            return backtrack(current_node + 1, current_assignment.copy(), honeypot_count)
        
        return backtrack(0, assignment.copy(), 0)
    
    def solve_heuristic(self) -> Dict[int, bool]:
        """Solve CSP using heuristic approach"""
        # Sort nodes by resource availability
        node_scores = []
        for i, node in enumerate(self.nodes):
            if node.can_host_honeypot(self.honeypot_requirements):
                score = sum(node.resources.values())
                node_scores.append((score, i))
        
        node_scores.sort(reverse=True)
        
        assignment = {i: False for i in range(len(self.nodes))}
        honeypot_count = 0
        
        for score, node_id in node_scores:
            if honeypot_count < self.max_honeypots:
                assignment[node_id] = True
                honeypot_count += 1
        
        return assignment

def create_network_topology(n_nodes: int = 10) -> nx.Graph:
    """Create a network topology"""
    G = nx.connected_watts_strogatz_graph(n_nodes, 4, 0.3)
    
    # Add resource attributes to nodes
    for node in G.nodes():
        G.nodes[node]['resources'] = {
            'cpu': random.randint(1, 4),
            'ram': random.randint(2, 16),
            'bandwidth': random.randint(100, 1000)
        }
        G.nodes[node]['pos'] = (random.random() * 100, random.random() * 100)
    
    return G