import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'model/fraud_model.joblib'
        self.scaler_path = 'model/scaler.joblib'

    def prepare_data(self, data):
        """
        Prepare data for training or prediction
        """
        # Select relevant features
        features = ['amount', 'hour', 'day', 'n_transactions_day', 'avg_amount_day']
        
        if 'is_fraud' in data.columns:
            X = data[features]
            y = data['is_fraud']
            return X, y
        return data[features]

    def train(self, data):
        """
        Train the fraud detection model
        """
        X, y = self.prepare_data(data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        self.save_model()
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'feature_importance': dict(zip(features, self.model.feature_importances_))
        }

    def predict(self, transaction_data):
        """
        Make predictions on new transactions
        """
        if self.model is None:
            self.load_model()
            
        # Prepare and scale the data
        X = self.prepare_data(transaction_data)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)[:, 1]
        
        return {
            'is_fraud': bool(prediction[0]),
            'fraud_probability': float(probability[0]),
            'risk_score': float(probability[0] * 100)
        }

    def save_model(self):
        """
        Save the trained model and scaler
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """
        Load the trained model and scaler
        """
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except FileNotFoundError:
            raise Exception("Model files not found. Please train the model first.")

def generate_sample_data(n_samples=1000):
    """
    Generate sample transaction data for training
    """
    np.random.seed(42)
    
    # Generate random transaction data
    data = {
        'amount': np.random.exponential(100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day': np.random.randint(1, 8, n_samples),
        'n_transactions_day': np.random.poisson(3, n_samples),
        'avg_amount_day': np.random.exponential(100, n_samples)
    }
    
    # Generate fraud labels (about 1% fraud rate)
    fraud_prob = np.zeros(n_samples)
    
    # Increase fraud probability for suspicious patterns
    fraud_prob += (data['amount'] > 500) * 0.3
    fraud_prob += (data['hour'] >= 22) * 0.2
    fraud_prob += (data['n_transactions_day'] > 5) * 0.2
    
    data['is_fraud'] = np.random.binomial(1, fraud_prob)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Generate sample data
    print("Generating sample transaction data...")
    sample_data = generate_sample_data(10000)
    
    # Train the model
    print("\nTraining fraud detection model...")
    model = FraudDetectionModel()
    results = model.train(sample_data)
    
    print(f"\nModel accuracy: {results['accuracy']:.4f}")
    print("\nFeature importance:")
    for feature, importance in results['feature_importance'].items():
        print(f"{feature}: {importance:.4f}")
    
    # Test prediction
    print("\nTesting prediction with a sample transaction...")
    sample_transaction = pd.DataFrame({
        'amount': [1000],
        'hour': [23],
        'day': [5],
        'n_transactions_day': [6],
        'avg_amount_day': [150]
    })
    
    prediction = model.predict(sample_transaction)
    print("\nPrediction results:")
    print(f"Is Fraud: {prediction['is_fraud']}")
    print(f"Fraud Probability: {prediction['fraud_probability']:.4f}")
    print(f"Risk Score: {prediction['risk_score']:.2f}%")
