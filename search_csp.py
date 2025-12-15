import heapq
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
import random
import time
from collections import deque

class AStarPlanner:
    """
    A* Search Algorithm for optimal honeypot placement planning
    """
    
    def __init__(self, network_graph: nx.Graph):
        self.graph = network_graph
        self.heuristic_cache = {}
    
    def heuristic(self, node1: int, node2: int) -> float:
        """
        Heuristic function for A* (Euclidean distance if coordinates exist,
        otherwise shortest path estimate)
        """
        if (node1, node2) in self.heuristic_cache:
            return self.heuristic_cache[(node1, node2)]
        
        # Try to use coordinates if available
        if 'pos' in self.graph.nodes[node1] and 'pos' in self.graph.nodes[node2]:
            pos1 = self.graph.nodes[node1]['pos']
            pos2 = self.graph.nodes[node2]['pos']
            distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        else:
            # Use degree difference as heuristic
            degree1 = self.graph.degree(node1)
            degree2 = self.graph.degree(node2)
            distance = abs(degree1 - degree2) / max(degree1, degree2, 1)
        
        self.heuristic_cache[(node1, node2)] = distance
        return distance
    
    def plan_path(self, start: int, goal: int, 
                  honeypot_nodes: List[int] = None) -> Tuple[List[int], float]:
        """
        Find optimal path using A* algorithm
        """
        if honeypot_nodes is None:
            honeypot_nodes = []
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes()}
        g_score[start] = 0
        
        f_score = {node: float('inf') for node in self.graph.nodes()}
        f_score[start] = self.heuristic(start, goal)
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return path, g_score[current]
            
            for neighbor in self.graph.neighbors(current):
                # Calculate tentative g score
                edge_weight = self.graph[current][neighbor].get('weight', 1.0)
                
                # Penalize paths through honeypots (they should be avoided for normal traffic)
                penalty = 2.0 if neighbor in honeypot_nodes else 1.0
                tentative_g_score = g_score[current] + edge_weight * penalty
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in [node for _, node in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [], float('inf')
    
    def reconstruct_path(self, came_from: Dict, current: int) -> List[int]:
        """
        Reconstruct path from came_from dictionary
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def find_optimal_placements(self, 
                               num_honeypots: int,
                               critical_nodes: List[int],
                               resource_constraints: Dict) -> List[int]:
        """
        Find optimal honeypot placements using A*
        """
        candidate_nodes = list(self.graph.nodes())
        
        # Score nodes based on centrality and proximity to critical nodes
        scores = []
        for node in candidate_nodes:
            # Calculate proximity score
            proximity_score = 0
            for critical in critical_nodes:
                try:
                    path_length = nx.shortest_path_length(self.graph, node, critical)
                    proximity_score += 1.0 / (path_length + 1)
                except:
                    proximity_score += 0
            
            # Calculate centrality score
            centrality = nx.degree_centrality(self.graph).get(node, 0)
            
            # Check resource constraints
            node_resources = self.graph.nodes[node].get('resources', {})
            feasible = all(
                node_resources.get(res, 0) >= resource_constraints[res]
                for res in resource_constraints
            )
            
            if feasible:
                total_score = proximity_score * 0.6 + centrality * 0.4
                scores.append((total_score, node))
        
        # Select top nodes
        scores.sort(reverse=True)
        selected_nodes = [node for _, node in scores[:num_honeypots]]
        
        return selected_nodes
    
    def best_first_search(self, start: int, goal: int) -> List[int]:
        """
        Best-First Search implementation
        """
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), start))
        
        came_from = {}
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic(neighbor, goal), neighbor))
        
        return []


class CSPOptimizer:
    """
    Constraint Satisfaction Problem solver for resource allocation
    """
    
    def __init__(self, nodes: List[int], constraints: Dict):
        self.nodes = nodes
        self.constraints = constraints
        self.domains = {}
        self.solution = None
        self.backtrack_count = 0
        
    def initialize_domains(self):
        """
        Initialize variable domains based on constraints
        """
        for node in self.nodes:
            node_info = self.constraints.get('node_info', {}).get(node, {})
            self.domains[node] = []
            
            # Generate possible resource allocations
            for cpu in range(1, node_info.get('max_cpu', 4) + 1):
                for mem in range(1, node_info.get('max_memory', 8) + 1):
                    for bw in range(1, node_info.get('max_bandwidth', 100) + 1):
                        allocation = {
                            'cpu': cpu,
                            'memory': mem,
                            'bandwidth': bw,
                            'node_id': node
                        }
                        if self.is_valid_allocation(allocation):
                            self.domains[node].append(allocation)
    
    def is_valid_allocation(self, allocation: Dict) -> bool:
        """
        Check if allocation satisfies basic constraints
        """
        # Check minimum requirements
        if allocation['cpu'] < 1 or allocation['memory'] < 1:
            return False
        
        # Check against global constraints
        global_constraints = self.constraints.get('global', {})
        
        if allocation['cpu'] > global_constraints.get('max_cpu_per_node', 8):
            return False
        if allocation['memory'] > global_constraints.get('max_memory_per_node', 16):
            return False
        
        return True
    
    def is_consistent(self, assignment: Dict) -> bool:
        """
        Check if current assignment is consistent with all constraints
        """
        # Check resource limits
        total_cpu = sum(alloc['cpu'] for alloc in assignment.values())
        total_memory = sum(alloc['memory'] for alloc in assignment.values())
        total_bandwidth = sum(alloc['bandwidth'] for alloc in assignment.values())
        
        global_constraints = self.constraints.get('global', {})
        
        if total_cpu > global_constraints.get('total_cpu', 32):
            return False
        if total_memory > global_constraints.get('total_memory', 64):
            return False
        if total_bandwidth > global_constraints.get('total_bandwidth', 1000):
            return False
        
        # Check node-specific constraints
        for node_id, allocation in assignment.items():
            node_info = self.constraints.get('node_info', {}).get(node_id, {})
            
            if allocation['cpu'] > node_info.get('max_cpu', 4):
                return False
            if allocation['memory'] > node_info.get('max_memory', 8):
                return False
        
        return True
    
    def select_unassigned_variable(self, assignment: Dict, use_mrv: bool = True) -> Optional[int]:
        """
        Select next unassigned variable using MRV (Minimum Remaining Values) heuristic
        """
        unassigned = [node for node in self.nodes if node not in assignment]
        
        if not unassigned:
            return None
        
        if use_mrv:
            # MRV: Choose variable with smallest domain
            return min(unassigned, key=lambda node: len(self.domains[node]))
        else:
            # Simple ordering
            return unassigned[0]
    
    def order_domain_values(self, node: int, assignment: Dict) -> List[Dict]:
        """
        Order domain values using least constraining value heuristic
        """
        values = self.domains[node].copy()
        
        # Sort by resource usage (least constraining first)
        values.sort(key=lambda x: (x['cpu'], x['memory'], x['bandwidth']))
        
        return values
    
    def backtrack(self, assignment: Dict, max_backtracks: int = 1000) -> Optional[Dict]:
        """
        Backtracking search with constraint propagation
        """
        self.backtrack_count += 1
        
        if self.backtrack_count > max_backtracks:
            return None
        
        # If assignment is complete
        if len(assignment) == len(self.nodes):
            return assignment
        
        # Select unassigned variable
        var = self.select_unassigned_variable(assignment)
        if var is None:
            return assignment
        
        # Try values in order
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            if self.is_consistent(new_assignment):
                result = self.backtrack(new_assignment, max_backtracks)
                if result is not None:
                    return result
        
        return None
    
    def solve(self, max_backtracks: int = 1000) -> Optional[Dict]:
        """
        Solve the CSP problem
        """
        self.initialize_domains()
        self.backtrack_count = 0
        
        initial_assignment = {}
        self.solution = self.backtrack(initial_assignment, max_backtracks)
        
        return self.solution
    
    def get_solution_cost(self) -> float:
        """
        Calculate cost of solution (lower is better)
        """
        if self.solution is None:
            return float('inf')
        
        total_cost = 0
        for allocation in self.solution.values():
            # Cost based on resource usage
            total_cost += allocation['cpu'] * 0.5
            total_cost += allocation['memory'] * 0.3
            total_cost += allocation['bandwidth'] * 0.01
        
        return total_cost
    
    def validate_constraints(self) -> Dict[str, bool]:
        """
        Validate that all constraints are satisfied
        """
        if self.solution is None:
            return {'valid': False, 'errors': ['No solution found']}
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check global constraints
        global_constraints = self.constraints.get('global', {})
        
        total_cpu = sum(alloc['cpu'] for alloc in self.solution.values())
        total_memory = sum(alloc['memory'] for alloc in self.solution.values())
        total_bandwidth = sum(alloc['bandwidth'] for alloc in self.solution.values())
        
        if 'total_cpu' in global_constraints and total_cpu > global_constraints['total_cpu']:
            validation['valid'] = False
            validation['errors'].append(f'Total CPU ({total_cpu}) exceeds limit ({global_constraints["total_cpu"]})')
        
        if 'total_memory' in global_constraints and total_memory > global_constraints['total_memory']:
            validation['valid'] = False
            validation['errors'].append(f'Total memory ({total_memory}) exceeds limit ({global_constraints["total_memory"]})')
        
        # Check node-specific constraints
        for node_id, allocation in self.solution.items():
            node_info = self.constraints.get('node_info', {}).get(node_id, {})
            
            if 'max_cpu' in node_info and allocation['cpu'] > node_info['max_cpu']:
                validation['valid'] = False
                validation['errors'].append(f'Node {node_id} CPU ({allocation["cpu"]}) exceeds limit ({node_info["max_cpu"]})')
        
        return validation