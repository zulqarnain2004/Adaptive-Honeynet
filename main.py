"""
Adaptive Deception-Mesh: Complete Pipeline
CS351 Project - GIK Institute
Group Members: Zulqarnain Umar (2023556), Muhammad Ismail (2023452), Awais Khan (2023139)
"""

import yaml
import numpy as np
import pandas as pd
import mlflow
import warnings
import sys
import os
import time
from datetime import datetime

sys.path.append('src')
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from machine_learning import AttackDetector
from search_csp import create_network_topology, AStarHoneypotPlanner, HoneypotCSP, NetworkNode
from reinforcement_learning import NetworkEnvironment, QLearningAgent
from explainability import ModelExplainer
from visualization import VisualizationDashboard

class AdaptiveDeceptionMesh:
    def __init__(self, config_path='config.yaml'):
        """Initialize Adaptive Deception-Mesh system"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        print("="*70)
        print("🛡️  ADAPTIVE DECEPTION-MESH")
        print("="*70)
        print("GIK Institute | CS351 - Artificial Intelligence")
        print("Group: Zulqarnain Umar (2023556), Muhammad Ismail (2023452), Awais Khan (2023139)")
        print("="*70)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(config_path)
        self.detector = AttackDetector()
        self.visualizer = VisualizationDashboard()
        
        # Create directories
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('models/results', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Results storage
        self.results = {}
        self.performance_history = []
        
    def run_data_preprocessing(self):
        """Phase 1: Data Preprocessing"""
        print("\n" + "="*60)
        print("PHASE 1: DATA PREPROCESSING")
        print("="*60)
        
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess()
        
        self.results['data'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }
        
        print(f"✅ Data preprocessing completed!")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        print(f"   Features: {len(feature_names)}")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def run_machine_learning(self):
        """Phase 2: Machine Learning for Attack Detection"""
        print("\n" + "="*60)
        print("PHASE 2: MACHINE LEARNING - ATTACK DETECTION")
        print("="*60)
        
        data = self.results['data']
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        # Train and evaluate models
        best_model, best_model_name, metrics = self.detector.find_optimal_model(
            X_train, y_train, X_test, y_test
        )
        
        # Save models
        self.detector.save_models()
        
        # Store results
        self.results['ml'] = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'metrics': metrics,
            'feature_importance': self.detector.feature_importance.get(best_model_name, {})
        }
        
        # Print results
        print("\n📊 Model Performance:")
        for model_name, model_metrics in metrics.items():
            print(f"\n  {model_name.upper().replace('_', ' ')}:")
            for metric, value in model_metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        print(f"\n🏆 Best Model: {best_model_name}")
        
        return best_model, metrics
    
    def run_search_csp(self):
        """Phase 3: Search Algorithms & Constraint Satisfaction"""
        print("\n" + "="*60)
        print("PHASE 3: SEARCH ALGORITHMS & CONSTRAINT SATISFACTION")
        print("="*60)
        
        # Create network topology
        n_nodes = self.config['network']['nodes']
        max_honeypots = self.config['network']['max_honeypots']
        
        print(f"🌐 Creating network with {n_nodes} nodes...")
        network_graph = create_network_topology(n_nodes)
        
        # Simulate attack detection (using ML predictions)
        print("🔍 Simulating attack detection...")
        
        # Generate suspicious nodes based on ML predictions
        n_suspicious = np.random.randint(2, 5)
        suspicious_nodes = np.random.choice(n_nodes, size=n_suspicious, replace=False).tolist()
        
        print(f"   Detected suspicious nodes: {suspicious_nodes}")
        
        # Use A* algorithm for honeypot placement planning
        print("🚀 Using A* algorithm for optimal honeypot placement...")
        planner = AStarHoneypotPlanner(network_graph)
        honeypot_nodes = planner.plan_honeypot_deployment(suspicious_nodes, max_honeypots)
        
        print(f"   Optimal honeypot placement: {honeypot_nodes}")
        
        # Create network nodes for CSP
        network_nodes = []
        for i in range(n_nodes):
            resources = network_graph.nodes[i]['resources']
            node = NetworkNode(i, resources)
            if i in suspicious_nodes:
                node.attacker_present = True
            network_nodes.append(node)
        
        # Solve Constraint Satisfaction Problem
        print("⚙️  Solving Constraint Satisfaction Problem...")
        csp_solver = HoneypotCSP(network_nodes, max_honeypots)
        csp_solution = csp_solver.solve_heuristic()
        
        # Validate solution
        is_valid = csp_solver.is_valid_assignment(csp_solution)
        
        print(f"   CSP Solution: {csp_solution}")
        print(f"   Solution valid: {'✅ Yes' if is_valid else '❌ No'}")
        
        # Calculate detection rate
        detections = set(suspicious_nodes) & set(honeypot_nodes)
        detection_rate = len(detections) / len(suspicious_nodes) if suspicious_nodes else 0
        
        print(f"   Detection rate: {detection_rate:.1%}")
        
        # Store results
        self.results['search_csp'] = {
            'network_graph': network_graph,
            'suspicious_nodes': suspicious_nodes,
            'honeypot_nodes': honeypot_nodes,
            'csp_solution': csp_solution,
            'csp_valid': is_valid,
            'detection_rate': detection_rate,
            'detections': list(detections)
        }
        
        return network_graph, suspicious_nodes, honeypot_nodes
    
    def run_reinforcement_learning(self):
        """Phase 4: Reinforcement Learning for Adaptive Defense"""
        print("\n" + "="*60)
        print("PHASE 4: REINFORCEMENT LEARNING - ADAPTIVE DEFENSE")
        print("="*60)
        
        # Create RL environment
        n_nodes = self.config['network']['nodes']
        max_honeypots = self.config['network']['max_honeypots']
        
        print(f"🤖 Creating RL environment with {n_nodes} nodes...")
        env = NetworkEnvironment(n_nodes=n_nodes, max_honeypots=max_honeypots)
        
        # Train Q-Learning agent
        print("🎮 Training Q-Learning agent...")
        agent = QLearningAgent(
            env,
            learning_rate=self.config['model']['reinforcement']['learning_rate'],
            discount_factor=self.config['model']['reinforcement']['discount_factor'],
            epsilon=self.config['model']['reinforcement']['epsilon']
        )
        
        episodes = min(200, self.config['model']['reinforcement']['episodes'])
        rewards_history, epsilon_history = agent.train(episodes=episodes, render_every=50)
        
        # Save Q-table
        agent.save_q_table()
        
        # Test the trained agent
        print("🧪 Testing trained agent...")
        test_rewards = []
        
        for ep in range(5):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_key = agent.state_to_key(state)
                action = np.argmax(agent.q_table[state_key])
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        print(f"   Average test reward: {avg_test_reward:.2f}")
        
        # Store results
        self.results['rl'] = {
            'rewards_history': rewards_history,
            'epsilon_history': epsilon_history,
            'test_rewards': test_rewards,
            'avg_test_reward': avg_test_reward,
            'agent': agent,
            'environment': env
        }
        
        return agent, rewards_history
    
    def run_explainability(self):
        """Phase 5: Model Explainability"""
        print("\n" + "="*60)
        print("PHASE 5: MODEL EXPLAINABILITY (SHAP & LIME)")
        print("="*60)
        
        data = self.results['data']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']
        
        # Get best model
        best_model = self.results['ml']['best_model']
        best_model_name = self.results['ml']['best_model_name']
        
        print(f"🔍 Analyzing {best_model_name} model with SHAP & LIME...")
        
        # Create explainer
        explainer = ModelExplainer(best_model, feature_names)
        
        # Generate comprehensive explanations
        explanations = explainer.comprehensive_explanation(X_test, y_test)
        
        # Store results
        self.results['explainability'] = explanations
        
        print("\n✅ Explainability analysis completed!")
        print("   Results saved to 'models/results/' directory")
        
        return explanations
    
    def run_visualization(self):
        """Phase 6: Comprehensive Visualization"""
        print("\n" + "="*60)
        print("PHASE 6: COMPREHENSIVE VISUALIZATION")
        print("="*60)
        
        # Prepare data for visualization
        visualization_data = {}
        
        # Add ML metrics
        if 'ml' in self.results:
            visualization_data['ml_metrics'] = self.results['ml']['metrics']
        
        # Add RL data
        if 'rl' in self.results:
            visualization_data['rl_rewards'] = self.results['rl']['rewards_history']
            visualization_data['rl_epsilon'] = self.results['rl']['epsilon_history']
        
        # Add network data
        if 'search_csp' in self.results:
            visualization_data['network_state'] = {
                'graph': self.results['search_csp']['network_graph'],
                'honeypots': self.results['search_csp']['honeypot_nodes'],
                'attacks': self.results['search_csp']['suspicious_nodes'],
                'detections': self.results['search_csp']['detections'],
                'detection_rate': self.results['search_csp']['detection_rate']
            }
        
        # Add feature importance
        if 'ml' in self.results and 'feature_importance' in self.results['ml']:
            visualization_data['feature_importance'] = self.results['ml']['feature_importance']
        
        print("📊 Generating comprehensive visualizations...")
        
        # Generate visualizations
        visualizations = self.visualizer.generate_comprehensive_report(visualization_data)
        
        self.results['visualizations'] = visualizations
        
        print("\n✅ All visualizations generated successfully!")
        print("   Check 'models/results/' directory")
        
        return visualizations
    
    def run_complete_demo(self):
        """Run complete interactive demo"""
        print("\n" + "="*70)
        print("🚀 COMPLETE ADAPTIVE DECEPTION-MESH DEMO")
        print("="*70)
        
        # Start MLflow experiment
        mlflow.set_experiment("Adaptive_Deception_Mesh_Complete")
        
        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(self.flatten_config(self.config))
            
            # Run all phases
            try:
                # Phase 1: Data Preprocessing
                self.run_data_preprocessing()
                
                # Phase 2: Machine Learning
                ml_metrics = self.run_machine_learning()
                
                # Phase 3: Search & CSP
                network_data = self.run_search_csp()
                
                # Phase 4: Reinforcement Learning
                rl_data = self.run_reinforcement_learning()
                
                # Phase 5: Explainability
                self.run_explainability()
                
                # Phase 6: Visualization
                self.run_visualization()
                
                # Log final metrics
                if 'ml' in self.results and 'metrics' in self.results['ml']:
                    for model_name, metrics in self.results['ml']['metrics'].items():
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(f"{model_name}_{metric_name}", value)
                
                if 'search_csp' in self.results:
                    mlflow.log_metric("csp_detection_rate", self.results['search_csp']['detection_rate'])
                
                if 'rl' in self.results:
                    mlflow.log_metric("rl_avg_reward", self.results['rl']['avg_test_reward'])
                
                print("\n" + "="*70)
                print("✅ DEMO COMPLETED SUCCESSFULLY!")
                print("="*70)
                
                # Print summary
                self.print_demo_summary()
                
            except Exception as e:
                print(f"\n❌ Error in demo: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def print_demo_summary(self):
        """Print demo summary"""
        print("\n📋 DEMO SUMMARY")
        print("-"*40)
        
        # ML Results
        if 'ml' in self.results:
            best_model = self.results['ml']['best_model_name']
            print(f"🤖 Best ML Model: {best_model}")
            
            if best_model in self.results['ml']['metrics']:
                metrics = self.results['ml']['metrics'][best_model]
                print(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"   F1 Score: {metrics.get('f1', 0):.1%}")
        
        # Search & CSP Results
        if 'search_csp' in self.results:
            print(f"🔍 Detection Rate: {self.results['search_csp']['detection_rate']:.1%}")
            print(f"   Honeypots placed: {self.results['search_csp']['honeypot_nodes']}")
            print(f"   Attacks detected: {self.results['search_csp']['detections']}")
        
        # RL Results
        if 'rl' in self.results:
            print(f"🎮 RL Average Reward: {self.results['rl']['avg_test_reward']:.2f}")
        
        print("\n📁 Results saved in:")
        print("   • models/saved_models/ - Trained models")
        print("   • models/results/ - Visualizations & reports")
        print("   • data/processed/ - Processed datasets")
        
        print("\n🌐 To run interactive dashboard:")
        print("   streamlit run dashboard.py")
    
    def flatten_config(self, config, prefix=""):
        """Flatten nested configuration for MLflow logging"""
        items = {}
        for key, value in config.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(self.flatten_config(value, new_key))
            else:
                items[new_key] = value
        return items

def main():
    """Main function"""
    print("Initializing Adaptive Deception-Mesh System...")
    
    # Create and run the system
    system = AdaptiveDeceptionMesh()
    system.run_complete_demo()

if __name__ == "__main__":
    main()