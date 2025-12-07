import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, model, feature_names: List[str], class_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Normal', 'Attack']
        
    def shap_analysis(self, X: np.ndarray, sample_size: int = 100) -> Dict:
        """Perform SHAP analysis on the model"""
        print("Performing SHAP analysis...")
        
        # Sample data for faster computation
        if len(X) > sample_size:
            X_sample = X[np.random.choice(len(X), sample_size, replace=False)]
        else:
            X_sample = X
        
        # Create explainer based on model type
        try:
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.Explainer(self.model.predict_proba, X_sample)
                shap_values = explainer(X_sample)
                
                # Get summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                                 show=False, max_display=15)
                plt.tight_layout()
                plt.savefig('models/results/shap_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Get feature importance
                shap_importance = np.abs(shap_values.values).mean(axis=0)
                feature_importance = dict(zip(self.feature_names, shap_importance))
                
                # Save top features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                print("\nTop 10 Important Features (SHAP):")
                for feature, importance in top_features:
                    print(f"{feature}: {importance:.4f}")
                
                return {
                    'shap_values': shap_values,
                    'feature_importance': feature_importance,
                    'top_features': top_features
                }
                
            else:
                print("Model doesn't support predict_proba, using KernelExplainer")
                explainer = shap.KernelExplainer(self.model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
                
                return {'shap_values': shap_values}
                
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return {}
    
    def lime_explanation(self, X: np.ndarray, instance_idx: int = 0, 
                        num_features: int = 10) -> Dict:
        """Generate LIME explanation for a specific instance"""
        print(f"\nGenerating LIME explanation for instance {instance_idx}...")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=self.class_names,
            verbose=True,
            mode='classification'
        )
        
        # Select instance
        instance = X[instance_idx]
        
        # Generate explanation
        if hasattr(self.model, 'predict_proba'):
            exp = explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features
            )
        else:
            exp = explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )
        
        # Save explanation as HTML
        exp.save_to_file('models/results/lime_explanation.html')
        
        # Get feature contributions
        feature_contributions = exp.as_list()
        
        print(f"\nLIME Explanation for instance {instance_idx}:")
        print("Class probabilities:", exp.predict_proba)
        print("\nTop contributing features:")
        for feature, contribution in feature_contributions:
            print(f"{feature}: {contribution:.4f}")
        
        return {
            'explanation': exp,
            'feature_contributions': feature_contributions,
            'predicted_class': exp.local_pred[0] if hasattr(exp, 'local_pred') else None,
            'confidence': exp.predict_proba
        }
    
    def permutation_importance_analysis(self, X: np.ndarray, y: np.ndarray, 
                                       n_repeats: int = 10, random_state: int = 42) -> Dict:
        """Calculate permutation importance"""
        print("\nCalculating permutation importance...")
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_n = min(20, len(importance_df))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(top_n), top_features['importance_mean'][::-1])
        plt.yticks(range(top_n), top_features['feature'][::-1])
        plt.xlabel('Permutation Importance')
        plt.title('Top Features by Permutation Importance')
        plt.tight_layout()
        plt.savefig('models/results/permutation_importance.png', dpi=300)
        plt.close()
        
        print("\nTop 10 Features by Permutation Importance:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        return {
            'importance_df': importance_df,
            'permutation_result': result
        }
    
    def feature_correlation_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Analyze feature correlations with target"""
        print("\nAnalyzing feature correlations...")
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = y
        
        # Calculate correlations
        correlations = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
        
        # Plot top correlations
        plt.figure(figsize=(10, 6))
        top_n = min(15, len(correlations))
        correlations.head(top_n).plot(kind='bar')
        plt.title(f'Top {top_n} Features Correlated with Target')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('models/results/feature_correlations.png', dpi=300)
        plt.close()
        
        print("\nTop 10 Features Correlated with Target:")
        for feature, corr in correlations.head(10).items():
            print(f"{feature}: {corr:.4f}")
        
        return {'correlations': correlations}
    
    def partial_dependence_analysis(self, X: np.ndarray, feature_indices: List[int],
                                   grid_resolution: int = 20) -> Dict:
        """Perform Partial Dependence Plot analysis"""
        print("\nPerforming Partial Dependence Analysis...")
        
        from sklearn.inspection import PartialDependenceDisplay
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select top features if indices not provided
        if not feature_indices:
            feature_indices = range(min(4, X.shape[1]))
        
        # Create PDP
        try:
            PartialDependenceDisplay.from_estimator(
                self.model, X,
                features=feature_indices,
                feature_names=self.feature_names,
                grid_resolution=grid_resolution,
                ax=ax
            )
            
            plt.tight_layout()
            plt.savefig('models/results/partial_dependence.png', dpi=300)
            plt.close()
            
            print("Partial Dependence Plots saved successfully")
            
            return {'pdp_created': True}
            
        except Exception as e:
            print(f"PDP analysis failed: {e}")
            return {'pdp_created': False}
    
    def comprehensive_explanation(self, X: np.ndarray, y: np.ndarray, 
                                 instance_idx: int = 0) -> Dict:
        """Generate comprehensive explanation using all methods"""
        print("=" * 60)
        print("GENERATING COMPREHENSIVE MODEL EXPLANATION")
        print("=" * 60)
        
        results = {}
        
        # Create results directory
        import os
        os.makedirs('models/results', exist_ok=True)
        
        # 1. SHAP Analysis
        results['shap'] = self.shap_analysis(X)
        
        # 2. LIME Explanation
        results['lime'] = self.lime_explanation(X, instance_idx)
        
        # 3. Permutation Importance
        results['permutation'] = self.permutation_importance_analysis(X, y)
        
        # 4. Feature Correlation
        results['correlation'] = self.feature_correlation_analysis(X, y)
        
        # 5. Partial Dependence
        results['pdp'] = self.partial_dependence_analysis(X)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict):
        """Generate summary report of explanations"""
        report_path = 'models/results/explanation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ADAPTIVE DECEPTION-MESH - MODEL EXPLANATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # SHAP Summary
            if 'shap' in results and 'top_features' in results['shap']:
                f.write("1. SHAP ANALYSIS - TOP FEATURES\n")
                f.write("-" * 40 + "\n")
                for feature, importance in results['shap']['top_features']:
                    f.write(f"{feature}: {importance:.4f}\n")
                f.write("\n")
            
            # LIME Summary
            if 'lime' in results and 'feature_contributions' in results['lime']:
                f.write("2. LIME EXPLANATION - FEATURE CONTRIBUTIONS\n")
                f.write("-" * 40 + "\n")
                for feature, contribution in results['lime']['feature_contributions']:
                    f.write(f"{feature}: {contribution:.4f}\n")
                f.write("\n")
            
            # Permutation Importance Summary
            if 'permutation' in results and 'importance_df' in results['permutation']:
                f.write("3. PERMUTATION IMPORTANCE - TOP FEATURES\n")
                f.write("-" * 40 + "\n")
                df = results['permutation']['importance_df'].head(10)
                for idx, row in df.iterrows():
                    f.write(f"{row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}\n")
                f.write("\n")
            
            # Correlation Summary
            if 'correlation' in results and 'correlations' in results['correlation']:
                f.write("4. FEATURE CORRELATION WITH TARGET\n")
                f.write("-" * 40 + "\n")
                correlations = results['correlation']['correlations'].head(10)
                for feature, corr in correlations.items():
                    f.write(f"{feature}: {corr:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("EXPLANATION FILES GENERATED:\n")
            f.write("1. shap_summary.png - SHAP summary plot\n")
            f.write("2. lime_explanation.html - LIME interactive explanation\n")
            f.write("3. permutation_importance.png - Permutation importance plot\n")
            f.write("4. feature_correlations.png - Feature correlation plot\n")
            f.write("5. partial_dependence.png - Partial dependence plots\n")
            f.write("6. explanation_summary.txt - This summary file\n")
        
        print(f"\nExplanation summary saved to: {report_path}")