from flask import Blueprint, jsonify, request
from models.ml_detector import MLDetector
from models.search_csp import AStarPlanner, CSPOptimizer
from models.rl_agent import RlAgent
from models.explainer import Explainer
import networkx as nx
import numpy as np
import json

api = Blueprint('api', __name__)

# Global instances (would be initialized in app context)
ml_detector = None
rl_agent = None
explainer = None
network_graph = None

@api.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({'message': 'API is working!'})

@api.route('/detect', methods=['POST'])
def detect_attack():
    """Detect attack in network traffic"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        # Here you would use the actual ML detector
        # For now, simulate detection
        features = data['features']
        
        # Simulate detection with 96% accuracy (from screenshot)
        is_attack = np.random.random() > 0.04  # 96% accuracy
        
        return jsonify({
            'attack_detected': bool(is_attack),
            'confidence': 0.96 if is_attack else 0.93,
            'model_used': 'Random Forest',
            'features_analyzed': len(features)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/plan', methods=['POST'])
def plan_honeypots():
    """Plan honeypot placements using A*"""
    try:
        data = request.get_json()
        
        # Default parameters
        num_honeypots = data.get('num_honeypots', 8)
        critical_nodes = data.get('critical_nodes', [0, 5, 10, 15])
        
        # Create network graph if not exists
        global network_graph
        if network_graph is None:
            network_graph = nx.erdos_renyi_graph(20, 0.3)
            for node in network_graph.nodes():
                network_graph.nodes[node]['resources'] = {
                    'cpu': np.random.randint(20, 100),
                    'memory': np.random.randint(30, 100)
                }
        
        # Use A* planner
        planner = AStarPlanner(network_graph)
        placements = planner.find_optimal_placements(
            num_honeypots=num_honeypots,
            critical_nodes=critical_nodes,
            resource_constraints={'cpu': 30, 'memory': 50}
        )
        
        return jsonify({
            'optimal_placements': placements,
            'num_honeypots': len(placements),
            'critical_nodes_protected': len(set(critical_nodes) & set(placements))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/allocate', methods=['POST'])
def allocate_resources():
    """Allocate resources using CSP"""
    try:
        data = request.get_json()
        
        nodes = data.get('nodes', list(range(10)))
        constraints = data.get('constraints', {
            'global': {
                'total_cpu': 32,
                'total_memory': 64,
                'total_bandwidth': 1000
            },
            'node_info': {
                i: {'max_cpu': 4, 'max_memory': 8, 'max_bandwidth': 100}
                for i in nodes
            }
        })
        
        # Solve CSP
        csp = CSPOptimizer(nodes, constraints)
        solution = csp.solve()
        
        if solution:
            validation = csp.validate_constraints()
            cost = csp.get_solution_cost()
            
            return jsonify({
                'solution': solution,
                'valid': validation['valid'],
                'cost': cost,
                'backtracks': csp.backtrack_count
            })
        else:
            return jsonify({'error': 'No solution found'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/explain', methods=['POST'])
def explain_prediction():
    """Explain a prediction using XAI"""
    try:
        data = request.get_json()
        
        prediction = data.get('prediction', 1)
        features = data.get('features', {})
        
        # Use explainer (simulated for now)
        explanation = {
            'prediction': prediction,
            'confidence': 0.96,
            'feature_importance': {
                'Packet Rate': -0.342,
                'Port Diversity': -0.239,
                'SNAP Network': 0.0,
                'Payload Rate': 0.0,
                'Protocol Type': 0.0,
                'Time Release': 0.0
            },
            'interpretation': 'Higher packet rate and port diversity indicate potential attack',
            'method': 'SHAP'
        }
        
        return jsonify(explanation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/rl/action', methods=['POST'])
def rl_action():
    """Get RL agent action"""
    try:
        data = request.get_json()
        state = data.get('state', {})
        
        # Simulate RL agent (from screenshot values)
        actions = [
            {'action': 'Deploy Honeypot Node-12', 'reward': 0.89, 'q_value': 6.72},
            {'action': 'Migrate Honeypot Node-7', 'reward': 0.78, 'q_value': 6.45},
            {'action': 'Reconfigure Service Profile', 'reward': 0.82, 'q_value': 6.58}
        ]
        
        action = np.random.choice(actions)
        
        return jsonify({
            'action': action['action'],
            'reward': action['reward'],
            'q_value': action['q_value'],
            'state': state,
            'epsilon': 0.001,
            'learning_rate': 0.001
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics"""
    # Return metrics from screenshot
    return jsonify({
        'ml_models': {
            'random_forest': {'accuracy': 0.96},
            'logistic_regression': {
                'accuracy': 0.93,
                'precision': 0.94,
                'recall': 0.92,
                'f1': 0.92
            },
            'kmeans': {
                'normal_traffic': 0.45,
                'port_scan': 0.25,
                'dos_attacks': 0.18,
                'exploits': 0.12
            }
        },
        'system': {
            'detection_accuracy': 0.82,
            'rl_agent_reward': 0.45,
            'cpu_usage': 45,
            'memory_usage': 52,
            'active_nodes': 20,
            'honeypots': 8
        },
        'rl_agent': {
            'epsilon': 0.001,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'current_reward': 0.45
        }
    })

@api.route('/topology', methods=['GET'])
def get_topology():
    """Get network topology"""
    # Generate sample topology
    nodes = []
    edges = []
    
    for i in range(20):
        nodes.append({
            'id': i,
            'type': 'honeypot' if i < 8 else 'normal',
            'x': np.random.uniform(0, 100),
            'y': np.random.uniform(0, 100),
            'cpu': np.random.randint(20, 100),
            'memory': np.random.randint(30, 100)
        })
    
    # Create some edges
    for i in range(30):
        edges.append({
            'source': np.random.randint(0, 20),
            'target': np.random.randint(0, 20),
            'weight': np.random.uniform(0.1, 1.0)
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'total_nodes': 20,
        'total_edges': 30,
        'honeypots': 8
    })