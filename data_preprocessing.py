import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import yaml
import os

class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self):
        """Load and preprocess UNSW-NB15 dataset"""
        df = pd.read_csv(self.config['data']['raw_path'])
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Drop unnecessary columns
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        return df
    
    def encode_categorical(self, df, columns):
        """Encode categorical variables"""
        for col in columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values found: {missing[missing > 0]}")
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Impute numeric columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            
            # For categorical, fill with mode
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        # Load data
        df = self.load_data()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables
        df = self.encode_categorical(df, categorical_cols)
        
        # Separate features and target
        if 'label' in df.columns:
            X = df.drop('label', axis=1)
            y = df['label']
        elif 'attack_cat' in df.columns:
            X = df.drop('attack_cat', axis=1)
            y = (df['attack_cat'] != 'Normal').astype(int)
        else:
            raise ValueError("No target column found")
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=self.config['data']['random_state'])
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE - X shape: {X_resampled.shape}, y shape: {y_resampled.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_resampled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_resampled,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_resampled
        )
        
        print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        
        # Save processed data
        processed_path = self.config['data']['processed_path']
        os.makedirs(processed_path, exist_ok=True)
        
        np.save(f"{processed_path}X_train.npy", X_train)
        np.save(f"{processed_path}X_test.npy", X_test)
        np.save(f"{processed_path}y_train.npy", y_train)
        np.save(f"{processed_path}y_test.npy", y_test)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, f"{processed_path}scaler.pkl")
        joblib.dump(self.label_encoders, f"{processed_path}label_encoders.pkl")
        
        return X_train, X_test, y_train, y_test, df.columns.tolist()
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        df = self.load_data()
        if 'label' in df.columns:
            return df.drop('label', axis=1).columns.tolist()
        elif 'attack_cat' in df.columns:
            return df.drop('attack_cat', axis=1).columns.tolist()
        return df.columns.tolist()