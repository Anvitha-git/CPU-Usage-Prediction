import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Define features as expected by the training script
FEATURES = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
# Controller kinds to simulate one-hot encoding
CONTROLLER_KINDS = ['Deployment', 'ReplicaSet', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']

def create_dummy_model():
    print("Creating dummy data...")
    # Create a small dummy dataset
    data = {
        'cpu_request': [100, 200, 150, 300, 100],
        'mem_request': [256, 512, 256, 1024, 128],
        'cpu_limit': [200, 400, 300, 600, 200],
        'mem_limit': [512, 1024, 512, 2048, 256],
        'runtime_minutes': [10, 20, 15, 30, 60],
        'controller_kind': ['Deployment', 'ReplicaSet', 'StatefulSet', 'DaemonSet', 'Job']
    }
    df = pd.DataFrame(data)
    
    # Target
    y = [0.1, 0.2, 0.15, 0.3, 0.05]
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['controller_kind'], drop_first=True)
    
    # Ensure all expected columns from one-hot encoding might be present in a real scenario
    # For a dummy model, we just need it to work with the features present in the dataframe
    
    X = df
    
    print("Training dummy model...")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    print("Saving model.pkl...")
    joblib.dump(model, "model.pkl")
    print("Done.")

if __name__ == "__main__":
    create_dummy_model()
