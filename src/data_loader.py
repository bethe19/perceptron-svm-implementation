import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataLoader:
    def __init__(self, file_path, target_column, positive_label=1, 
                 negative_label=0, scale_features=True, drop_columns=None):
        self.file_path = file_path
        self.target_column = target_column
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.scale_features = scale_features
        self.drop_columns = drop_columns
        self.scaler = None
        self.X_scaled = None
        self.X_real = None
        self.y = None
        self.data = None
    
    def load(self):
        self.data = pd.read_csv(self.file_path)
        
        if self.drop_columns:
            self.data = self.data.drop(columns=self.drop_columns)
        
        self.y = self.data[self.target_column].map({
            self.positive_label: 1, 
            self.negative_label: 0
        }).values
        
        self.X_real = self.data.drop(columns=[self.target_column]).values
        
        if self.scale_features:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X_real)
        else:
            self.X_scaled = self.X_real
        
        return self.X_scaled, self.y, self.X_real
    
    def transform(self, X):
        if self.scaler is None:
            if self.scale_features:
                raise ValueError("Scaler has not been fitted yet. Call load() first.")
            else:
                return X
        
        return self.scaler.transform(X)
    
    def get_feature_names(self):
        if self.data is None:
            raise ValueError("Data has not been loaded yet. Call load() first.")
        
        return [col for col in self.data.columns if col != self.target_column]
    
    def get_dataset_info(self):
        if self.data is None:
            raise ValueError("Data has not been loaded yet. Call load() first.")
        
        return {
            'n_samples': len(self.y),
            'n_features': self.X_real.shape[1],
            'n_positive': np.sum(self.y == 1),
            'n_negative': np.sum(self.y == 0),
            'class_balance': np.sum(self.y == 1) / len(self.y) * 100,
            'feature_names': self.get_feature_names()
        }