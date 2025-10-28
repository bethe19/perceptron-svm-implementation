import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(
    file_path, 
    target_column, 
    positive_label=1, 
    negative_label=0, 
    scale_features=True, 
    drop_columns=None
):
    data = pd.read_csv(file_path)
    
    if drop_columns:
        data = data.drop(columns=drop_columns)
    
    y = data[target_column].map({positive_label: 1, negative_label: 0}).values
    X_real = data.drop(columns=[target_column]).values
    
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_real)
    else:
        X_scaled = X_real
    
    return X_scaled, y, X_real