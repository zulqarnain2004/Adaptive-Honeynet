import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append('src')

from data_preprocessing import DataPreprocessor
from machine_learning import AttackDetector
from search_csp import NetworkNode, HoneypotCSP, create_network_topology
from reinforcement_learning import NetworkEnvironment

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
    
    def test_load_data(self):
        """Test data loading"""
        df = self.preprocessor.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        print(f"Data shape: {df.shape}")
    
    def test_preprocess(self):
        """Test preprocessing pipeline"""
        X_train, X_test, y_train, y_test, features = self.preprocessor.preprocess()
        
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(y_train), 0)
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

class TestMachineLearning(unittest.TestCase):
    def setUp(self):
        self.detector = AttackDetector()
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_random_forest(self):
        """Test Random Forest training"""
        rf = self.detector.train_random_forest(self.X_train, self.y_train)
        self.assertIsNotNone(rf)
        
        predictions = rf.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        print(f"RF predictions: {predictions[:5]}")
    
    def test_logistic_regression(self):
        """Test Logistic Regression training"""
        lr = self.detector.train_logistic_regression(self.X_train, self.y_train)
        self.assertIsNotNone(lr)
        
        predictions = lr.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        print(f"LR predictions: {predictions[:5]}")
    
    def test_xgboost(self):
        """Test XGBoost training"""
        xgb = self.detector.train_xgboost(self.X_train, self.y_train)
        self.assertIsNotNone(xgb)
        
        predictions = xgb.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        print(f"XGBoost predictions: {predictions[:5]}")

class TestSearchCSP(unittest.TestCase):
    def test_network_node(self):
        """Test NetworkNode class"""
        resources = {'cpu': 4, 'ram': 16, 'bandwidth': 1000}
        node = NetworkNode(1, resources)
        
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.resources['cpu'], 4)
        self.assertFalse(node.honeypot)
        
        # Test resource allocation
        requirements = {'cpu': 2, 'ram': 8, 'bandwidth': 500}
        self.assertTrue(node.can_host_honeypot(requirements))
        
        node.allocate_resources(requirements)
        self.assertTrue(node.honeypot)
        self.assertEqual(node.resources['cpu'], 2)  # 4 - 2 = 2
    
    def test_honeypot_csp(self):
        """Test CSP solver"""
        # Create network nodes
        nodes = [
            NetworkNode(0, {'cpu': 4, 'ram': 16, 'bandwidth': 1000}),
            NetworkNode(1, {'cpu': 2, 'ram': 8, 'bandwidth': 500}),
            NetworkNode(2, {'cpu': 1, 'ram': 4, 'bandwidth': 200}),
        ]
        
        csp = HoneypotCSP(nodes, max_honeypots=2)
        solution = csp.solve_heuristic()
        
        self.assertIsInstance(solution, dict)
        self.assertEqual(len(solution), 3)
        print(f"CSP solution: {solution}")
    
    def test_network_topology(self):
        """Test network creation"""
        G = create_network_topology(5)
        
        self.assertEqual(G.number_of_nodes(), 5)
        self.assertGreater(G.number_of_edges(), 0)
        
        # Check node attributes
        for node in G.nodes():
            self.assertIn('resources', G.nodes[node])
            self.assertIn('cpu', G.nodes[node]['resources'])
            print(f"Node {node} resources: {G.nodes[node]['resources']}")

class TestReinforcementLearning(unittest.TestCase):
    def test_environment(self):
        """Test RL environment"""
        env = NetworkEnvironment(n_nodes=5, max_honeypots=2)
        
        # Test reset
        state = env.reset()
        self.assertEqual(len(state), 10)  # 2 * n_nodes
        
        # Test step
        action = 1  # Toggle honeypot at node 0
        next_state, reward, done, info = env.step(action)
        
        self.assertEqual(len(next_state), 10)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        
        print(f"Initial state: {state[:5]}...")
        print(f"Next state: {next_state[:5]}...")
        print(f"Reward: {reward}, Done: {done}")

def run_all_tests():
    """Run all tests"""
    print("Running Adaptive Deception-Mesh Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestMachineLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchCSP))
    suite.addTests(loader.loadTestsFromTestCase(TestReinforcementLearning))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result

if __name__ == '__main__':
    run_all_tests()