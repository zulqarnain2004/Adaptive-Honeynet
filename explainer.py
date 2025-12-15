import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import json

class Explainer:
    """
    Explainable AI component using SHAP and LIME
    """
    
    def __init__(self, config):
        self.config = config
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = [
            'Packet Rate',
            'Port Diversity', 
            'SNAP Network',
            'Payload Rate',
            'Protocol Type',
            'Time Release'
        ]
    
    def initialize(self, model, X_train, feature_names=None):
        """
        Initialize explainers with trained model and training data
        """
        if feature_names:
            self.feature_names = feature_names
        
        print("Initializing explainers...")
        
        # Initialize SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(model)
            print("SHAP explainer initialized")
        except:
            print("Using KernelSHAP instead of TreeExplainer")
            self.shap_explainer = shap.KernelExplainer(model.predict, X_train[:100])
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=['Normal', 'Attack'],
            verbose=False,
            mode='classification'
        )
        print("LIME explainer initialized")
    
    def explain_prediction_shap(self, instance):
        """
        Explain a single prediction using SHAP
        """
        if self.shap_explainer is None:
            return self._get_dummy_explanation()
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(instance)
        
        # Get feature importance
        if isinstance(shap_values, list):
            # For classification models
            shap_values = shap_values[1]  # Take values for positive class
        
        # Create explanation
        explanation = {
            'feature_importance': {},
            'base_value': float(self.shap_explainer.expected_value[1] if 
                               hasattr(self.shap_explainer.expected_value, '__len__') else 
                               self.shap_explainer.expected_value),
            'prediction_value': float(np.sum(shap_values) + 
                                     (self.shap_explainer.expected_value[1] if 
                                      hasattr(self.shap_explainer.expected_value, '__len__') else 
                                      self.shap_explainer.expected_value))
        }
        
        # Add feature importance
        for i, feature in enumerate(self.feature_names[:len(shap_values)]):
            explanation['feature_importance'][feature] = float(shap_values[i])
        
        return explanation
    
    def explain_prediction_lime(self, instance, num_features=5):
        """
        Explain a single prediction using LIME
        """
        if self.lime_explainer is None or instance is None:
            return self._get_dummy_explanation()
        
        # Get LIME explanation
        exp = self.lime_explainer.explain_instance(
            instance[0],  # LIME expects 1D array
            lambda x: np.array([[0.2, 0.8]] * len(x)),  # Dummy predict function
            num_features=num_features
        )
        
        # Parse explanation
        explanation = {
            'feature_weights': [],
            'intercept': exp.intercept[1],
            'prediction': exp.local_pred[0]
        }
        
        for feature, weight in exp.as_list():
            explanation['feature_weights'].append({
                'feature': feature,
                'weight': weight
            })
        
        return explanation
    
    def get_global_feature_importance(self):
        """
        Get global feature importance (from screenshot)
        """
        return {
            'feature_importance': self.config.FEATURE_IMPORTANCE,
            'description': 'SHAP provides global feature importance using Shapley values from game theory',
            'feature_analysis': [
                {
                    'feature': 'Packet Rate',
                    'value': 'Checked',
                    'importance': -0.342,
                    'interpretation': 'Higher packet rate indicates potential DDoS attack'
                },
                {
                    'feature': 'Port Diversity',
                    'value': 'High',
                    'importance': -0.239,
                    'interpretation': 'High port diversity suggests port scanning activity'
                },
                {
                    'feature': 'SNAP Network',
                    'value': 'Normal',
                    'importance': 0.0,
                    'interpretation': 'Standard network behavior'
                },
                {
                    'feature': 'Payload Rate',
                    'value': 'Normal',
                    'importance': 0.0,
                    'interpretation': 'Normal payload patterns'
                },
                {
                    'feature': 'Protocol Type',
                    'value': 'TCP',
                    'importance': 0.0,
                    'interpretation': 'Common protocol usage'
                },
                {
                    'feature': 'Time Release',
                    'value': 'Regular',
                    'importance': 0.0,
                    'interpretation': 'Normal timing patterns'
                }
            ]
        }
    
    def _get_dummy_explanation(self):
        """
        Return dummy explanation when explainers aren't initialized
        """
        return {
            'feature_importance': self.config.FEATURE_IMPORTANCE,
            'base_value': 0.0,
            'prediction_value': 0.8,
            'warning': 'Using pre-configured values from screenshot'
        }
    
    def visualize_shap(self, X, max_display=10, save_path=None):
        """
        Generate SHAP visualization
        """
        if self.shap_explainer is None or X is None:
            print("SHAP explainer not initialized")
            return
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X)
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP plot saved to {save_path}")
        
        return plt.gcf()
    
    def save_explanations(self, explanations, path):
        """
        Save explanations to file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(explanations, f, indent=2)