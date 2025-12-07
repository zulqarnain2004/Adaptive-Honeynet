import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class AttackDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           n_estimators: int = 100, max_depth: int = 10) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        self.feature_importance['random_forest'] = rf.feature_importances_
        return rf
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 C: float = 1.0) -> LogisticRegression:
        """Train Logistic Regression classifier"""
        lr = LogisticRegression(
            C=C,
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        return lr
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     n_estimators: int = 100, max_depth: int = 6) -> xgb.XGBClassifier:
        """Train XGBoost classifier"""
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        self.feature_importance['xgboost'] = xgb_model.feature_importances_
        return xgb_model
    
    def train_kmeans(self, X: np.ndarray, n_clusters: int = 5) -> KMeans:
        """Train KMeans clustering for anomaly detection"""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(X)
        self.models['kmeans'] = kmeans
        return kmeans
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "") -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      model_type: str = 'random_forest', 
                      n_splits: int = 5) -> Dict:
        """Perform cross-validation"""
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
        
        return {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
    
    def find_optimal_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Tuple:
        """Find the best performing model"""
        best_score = 0
        best_model_name = ""
        best_model = None
        all_metrics = {}
        
        # Start MLflow experiment
        mlflow.set_experiment("Adaptive_Deception_Mesh")
        
        # Train and evaluate each model
        with mlflow.start_run():
            # Random Forest
            rf_model = self.train_random_forest(X_train, y_train)
            rf_metrics = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
            all_metrics['random_forest'] = rf_metrics
            
            mlflow.log_metrics({f"rf_{k}": v for k, v in rf_metrics.items()})
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            
            if rf_metrics['f1'] > best_score:
                best_score = rf_metrics['f1']
                best_model_name = "random_forest"
                best_model = rf_model
            
            # Logistic Regression
            lr_model = self.train_logistic_regression(X_train, y_train)
            lr_metrics = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
            all_metrics['logistic_regression'] = lr_metrics
            
            mlflow.log_metrics({f"lr_{k}": v for k, v in lr_metrics.items()})
            mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
            
            if lr_metrics['f1'] > best_score:
                best_score = lr_metrics['f1']
                best_model_name = "logistic_regression"
                best_model = lr_model
            
            # XGBoost
            xgb_model = self.train_xgboost(X_train, y_train)
            xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            all_metrics['xgboost'] = xgb_metrics
            
            mlflow.log_metrics({f"xgb_{k}": v for k, v in xgb_metrics.items()})
            mlflow.xgboost.log_model(xgb_model, "xgboost_model")
            
            if xgb_metrics['f1'] > best_score:
                best_score = xgb_metrics['f1']
                best_model_name = "xgboost"
                best_model = xgb_model
            
            # KMeans clustering (for anomaly detection)
            kmeans_model = self.train_kmeans(X_train, n_clusters=5)
            self.models['kmeans'] = kmeans_model
            
            # Log best model
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_f1_score", best_score)
        
        print(f"\nBest Model: {best_model_name} with F1 Score: {best_score:.4f}")
        
        self.best_model = best_model
        return best_model, best_model_name, all_metrics
    
    def detect_attacks(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Detect attacks using trained model"""
        if model_name:
            model = self.models.get(model_name, self.best_model)
        else:
            model = self.best_model
        
        if model is None:
            raise ValueError("No trained model available")
        
        return model.predict(X)
    
    def detect_anomalies(self, X: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect anomalies using clustering"""
        if 'kmeans' not in self.models:
            raise ValueError("KMeans model not trained")
        
        kmeans = self.models['kmeans']
        distances = kmeans.transform(X)
        min_distances = np.min(distances, axis=1)
        
        # Anomaly if distance > threshold * mean distance
        anomaly_threshold = threshold * np.mean(min_distances)
        anomalies = min_distances > anomaly_threshold
        
        return anomalies.astype(int)
    
    def save_models(self, path: str = "models/saved_models/"):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'xgboost':
                joblib.dump(model, f"{path}{name}.pkl")
            else:
                joblib.dump(model, f"{path}{name}.pkl")
        
        print(f"Models saved to {path}")
    
    def load_models(self, path: str = "models/saved_models/"):
        """Load trained models"""
        import os
        for name in ['random_forest', 'logistic_regression', 'xgboost', 'kmeans']:
            file_path = f"{path}{name}.pkl"
            if os.path.exists(file_path):
                self.models[name] = joblib.load(file_path)
                if name in ['random_forest', 'xgboost']:
                    self.feature_importance[name] = self.models[name].feature_importances_
        
        self.best_model = self.models.get('xgboost', 
                                         self.models.get('random_forest', 
                                                        self.models.get('logistic_regression')))
        print("Models loaded successfully")