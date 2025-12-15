import unittest
import json
from app import create_app
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAdaptiveDeceptionMeshAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_index_page(self):
        """Test main index page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('name', data)
        self.assertEqual(data['name'], 'Adaptive Deception Mesh')
    
    def test_system_status(self):
        """Test system status endpoint"""
        response = self.client.get('/api/v1/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check required fields
        required_fields = ['system_status', 'nodes', 'honeypots', 
                          'detection_accuracy', 'rl_agent_reward']
        for field in required_fields:
            self.assertIn(field, data)
    
    def test_ml_metrics(self):
        """Test ML metrics endpoint"""
        response = self.client.get('/api/v1/ml-metrics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check ML model data
        self.assertIn('random_forest', data)
        self.assertIn('logistic_regression', data)
        self.assertIn('kmeans_clustering', data)
    
    def test_rl_metrics(self):
        """Test RL metrics endpoint"""
        response = self.client.get('/api/v1/rl-metrics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check RL agent data
        self.assertIn('epsilon', data)
        self.assertIn('learning_rate', data)
        self.assertIn('gamma', data)
        self.assertIn('reward', data)
    
    def test_csp_constraints(self):
        """Test CSP constraints endpoint"""
        response = self.client.get('/api/v1/csp-constraints')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check CSP data
        self.assertIn('active_constraints', data)
        self.assertIn('resource_distribution', data)
        self.assertIn('current_values', data)
    
    def test_explainability(self):
        """Test explainability endpoint"""
        response = self.client.get('/api/v1/explainability')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check explainability data
        self.assertIn('shap_feature_importance', data)
        self.assertIn('feature_analysis', data)
    
    def test_simulation_control(self):
        """Test simulation control endpoints"""
        # Test start simulation
        response = self.client.post('/api/v1/simulation/start')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('success', data)
        
        # Test stop simulation
        response = self.client.post('/api/v1/simulation/stop')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('success', data)
    
    def test_network_topology(self):
        """Test network topology endpoint"""
        response = self.client.get('/api/v1/network/topology')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check topology data structure
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        self.assertIn('honeypots', data)
    
    def test_attack_detection(self):
        """Test attack detection endpoint"""
        test_data = {
            'network_data': {
                'src_ip': '192.168.1.100',
                'dst_ip': '10.0.0.1',
                'packet_count': 1000,
                'duration': 10.5
            }
        }
        
        response = self.client.post('/api/v1/detect',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertIn('predictions', data)
        self.assertIn('attack_detected', data)
    
    def test_invalid_detection_request(self):
        """Test attack detection with invalid data"""
        response = self.client.post('/api/v1/detect',
                                   data=json.dumps({}),
                                   content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_all_endpoints_exist(self):
        """Test that all documented endpoints exist"""
        endpoints = [
            ('/', 'GET'),
            ('/health', 'GET'),
            ('/api/v1/status', 'GET'),
            ('/api/v1/ml-metrics', 'GET'),
            ('/api/v1/rl-metrics', 'GET'),
            ('/api/v1/csp-constraints', 'GET'),
            ('/api/v1/explainability', 'GET'),
            ('/api/v1/network/topology', 'GET'),
            ('/api/v1/simulation/start', 'POST'),
            ('/api/v1/simulation/stop', 'POST'),
            ('/api/v1/detect', 'POST')
        ]
        
        for endpoint, method in endpoints:
            if method == 'GET':
                response = self.client.get(endpoint)
            else:
                response = self.client.post(endpoint)
            
            # Should not get 404
            self.assertNotEqual(response.status_code, 404, 
                              f"Endpoint {endpoint} returned 404")

if __name__ == '__main__':
    unittest.main()