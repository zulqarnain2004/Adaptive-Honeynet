import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Data preprocessing for cybersecurity datasets
    """
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
    
    def preprocess_unswnb15(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess UNSW-NB15 dataset
        """
        print("Preprocessing UNSW-NB15 dataset...")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Select features (based on importance from screenshot)
        feature_columns = self.select_features(df)
        
        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical(df, feature_columns)
        
        # Step 4: Handle class imbalance
        df_balanced = self.handle_class_imbalance(df_encoded)
        
        # Step 5: Split features and labels
        X = df_balanced[feature_columns].values
        y = df_balanced['label'].values if 'label' in df_balanced.columns else np.zeros(len(df_balanced))
        
        # Step 6: Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Preprocessing complete: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Found {missing.sum()} missing values")
            
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> list:
        """
        Select important features for attack detection
        Based on SHAP importance from screenshot
        """
        # Priority features from screenshot
        priority_features = [
            # From screenshot: Packet Rate, Port Diversity, etc.
            'spkts', 'dpkts',  # Packet rate proxies
            'sport', 'dsport',  # Port diversity proxies
            'dur',              # Time features
            'sbytes', 'dbytes', # Payload rate
            'proto',            # Protocol type
            'stime', 'ltime',   # Time release
        ]
        
        # Additional important features from UNSW-NB15
        additional_features = [
            'sttl', 'dttl',     # TTL values
            'sload', 'dload',   # Load rates
            'swin', 'dwin',     # Window sizes
            'sjit', 'djit',     # Jitter
            'smeansz', 'dmeansz', # Mean packet size
            'ct_state_ttl',     # Connection state TTL
            'ct_srv_src',       # Connection service source
            'ct_srv_dst',       # Connection service destination
            'is_sm_ips_ports',  # Same IP/ports
            'ct_flw_http_mthd', # HTTP methods
        ]
        
        # Check which features exist in the dataset
        available_features = []
        for feature in priority_features + additional_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # Add all numeric features if we don't have enough
        if len(available_features) < 10:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove label column if it exists
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            available_features = list(set(available_features + numeric_cols[:20]))
        
        self.feature_columns = available_features
        print(f"Selected {len(available_features)} features")
        
        return available_features
    
    def encode_categorical(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Encode categorical features
        """
        df_encoded = df.copy()
        
        for col in feature_columns:
            if df_encoded[col].dtype == 'object':
                # Create or reuse label encoder
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def handle_class_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle class imbalance using resampling
        """
        if 'label' not in df.columns:
            return df
        
        # Check class distribution
        class_counts = df['label'].value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        # Check if imbalance is severe
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        if max_count / min_count > 2:  # If imbalance is more than 2:1
            print("Handling class imbalance...")
            
            # Use oversampling for minority class
            from sklearn.utils import resample
            
            dfs = []
            for class_label in class_counts.index:
                df_class = df[df['label'] == class_label]
                
                if len(df_class) < max_count:
                    # Oversample minority class
                    df_class_oversampled = resample(
                        df_class,
                        replace=True,
                        n_samples=max_count,
                        random_state=self.config.RANDOM_STATE
                    )
                    dfs.append(df_class_oversampled)
                else:
                    # Keep majority class as is or undersample
                    df_class_sampled = resample(
                        df_class,
                        replace=False,
                        n_samples=max_count,
                        random_state=self.config.RANDOM_STATE
                    )
                    dfs.append(df_class_sampled)
            
            df_balanced = pd.concat(dfs)
            print(f"After balancing: {df_balanced['label'].value_counts().to_dict()}")
            return df_balanced
        
        return df
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Split data into train and test sets
        """
        return train_test_split(
            X, y,
            test_size=1 - self.config.TRAIN_TEST_SPLIT,
            random_state=self.config.RANDOM_STATE,
            stratify=y if len(np.unique(y)) > 1 else None
        )
    
    def preprocess_for_training(self, df: pd.DataFrame) -> dict:
        """
        Complete preprocessing pipeline for training
        """
        # Preprocess data
        X, y = self.preprocess_unswnb15(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_columns,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
    
    def preprocess_single_sample(self, sample: dict) -> np.ndarray:
        """
        Preprocess a single sample for prediction
        """
        # Convert to DataFrame
        df_sample = pd.DataFrame([sample])
        
        # Handle missing values
        df_sample = self.handle_missing_values(df_sample)
        
        # Encode categorical features
        for col in self.feature_columns:
            if col in df_sample.columns and col in self.label_encoders:
                try:
                    df_sample[col] = self.label_encoders[col].transform([str(df_sample[col].iloc[0])])[0]
                except:
                    # If value not seen before, use default
                    df_sample[col] = 0
        
        # Select features
        X = np.zeros((1, len(self.feature_columns)))
        for i, feature in enumerate(self.feature_columns):
            if feature in df_sample.columns:
                X[0, i] = df_sample[feature].iloc[0]
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return X_scaled