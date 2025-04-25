import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATASET_PATH = 'model/dataset/creditcard.csv'
MODEL_PATH = 'model/fraud_model.joblib'
SCALER_PATH = 'model/scaler.joblib'

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please download from Kaggle and place it there.")
    data = pd.read_csv(DATASET_PATH)
    return data

def preprocess_data(data):
    # Features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Scale 'Amount' and 'Time' features
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
    
    return X, y, scaler

def train_model():
    data = load_data()
    X, y, scaler = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model and scaler saved to {MODEL_PATH} and {SCALER_PATH}")

if __name__ == '__main__':
    train_model()
