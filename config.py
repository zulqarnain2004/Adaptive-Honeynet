import os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class Config:
    # Basic Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-adaptive-deception-mesh'
    DEBUG = False
    TESTING = False
    
    # API Configuration
    API_TITLE = 'Adaptive Deception Mesh API'
    API_VERSION = 'v1'
    OPENAPI_VERSION = '3.0.2'
    
    # ML Configuration
    MODEL_DIR = 'models/saved_models'
    DATA_DIR = 'data'
    LOG_DIR = 'logs'
    
    # UNSW-NB15 Dataset Configuration
    DATASET_PATH = os.path.join(DATA_DIR, 'unswnb15_data.csv')
    
    # Model Parameters
    TRAIN_TEST_SPLIT = 0.7
    RANDOM_STATE = 42
    MAX_ITERATIONS = 1000
    
    # RL Configuration (from screenshot)
    RL_GAMMA = 0.99  # Discount factor from screenshot
    RL_LEARNING_RATE = 0.001  # Learning rate from screenshot
    RL_EPSILON = 0.1
    RL_MAX_EPISODES = 1000
    RL_REWARD_PROGRESSION = [0.7, 0.5, 0.2]  # From screenshot
    
    # Network Configuration (from screenshot)
    MAX_HONEYPOTS = 8  # From screenshot
    NETWORK_NODES = 20  # From screenshot
    NETWORK_EDGES = 30
    
    # CSP Configuration (from screenshot)
    CSP_TIMEOUT = 60
    MAX_BACKTRACKS = 1000
    CPU_THRESHOLD = 95  # From screenshot constraint
    MEMORY_THRESHOLD = 80  # From screenshot constraint
    MIN_HONEYPOTS = 4  # From screenshot constraint
    
    # ML Model Metrics (from screenshot)
    RANDOM_FOREST_ACCURACY = 0.96
    LOGISTIC_REGRESSION_ACCURACY = 0.93
    PRECISION = 0.94
    RECALL = 0.97
    F1_SCORE = 0.93
    DETECTION_ACCURACY = 0.82  # System detection accuracy
    RL_AGENT_REWARD = 0.45  # From screenshot
    
    # K-Means Clustering (from screenshot)
    CLUSTER_DISTRIBUTION = {
        'Normal Traffic': 0.45,
        'Port Scan': 0.25,
        'DoS Attacks': 0.18,
        'Exploits': 0.12
    }
    
    # SHAP Feature Importance (from screenshot)
    FEATURE_IMPORTANCE = {
        'Packet Rate': -0.342,
        'Port Diversity': -0.239,
        'SNAP Network': 0.0,
        'Payload Rate': 0.0,
        'Protocol Type': 0.0,
        'Time Release': 0.0
    }

class DevelopmentConfig(Config):
    DEBUG = True
    DEVELOPMENT = True

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}