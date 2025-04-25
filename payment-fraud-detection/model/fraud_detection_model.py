import numpy as np
import pandas as pd
from datetime import datetime

class SimpleFraudDetector:
    def __init__(self):
        # Define threshold values for different rules
        self.amount_threshold = 1000.0
        self.night_hours = set(range(23, 24)) | set(range(0, 5))
        self.max_daily_transactions = 5
        self.suspicious_countries = {'RU', 'NG', 'CN'}

    def calculate_risk_score(self, transaction):
        """
        Calculate risk score based on simple rules
        Returns a score between 0 and 100
        """
        risk_score = 0
        
        # Check transaction amount
        if transaction['amount'] > self.amount_threshold:
            risk_score += 30
        elif transaction['amount'] > self.amount_threshold * 0.7:
            risk_score += 15

        # Check transaction time
        hour = transaction.get('hour', datetime.now().hour)
        if hour in self.night_hours:
            risk_score += 20

        # Check transaction frequency
        daily_transactions = transaction.get('n_transactions_day', 1)
        if daily_transactions > self.max_daily_transactions:
            risk_score += 25

        # Check location if available
        country = transaction.get('country', '')
        if country in self.suspicious_countries:
            risk_score += 25

        # Normalize score to 0-100 range
        return min(100, risk_score)

    def predict(self, transaction_data):
        """
        Predict if a transaction is fraudulent
        Returns a dictionary with prediction details
        """
        risk_score = self.calculate_risk_score(transaction_data)
        
        return {
            'is_fraud': risk_score > 70,
            'risk_score': risk_score,
            'risk_level': 'High' if risk_score > 70 else 'Medium' if risk_score > 30 else 'Low',
            'timestamp': datetime.now().isoformat()
        }

    def get_rules_description(self):
        """
        Return the rules used for fraud detection
        """
        return {
            'amount_threshold': f'Transactions above ${self.amount_threshold:,.2f} are considered high risk',
            'time_check': 'Transactions between 11 PM and 5 AM are flagged',
            'frequency_check': f'More than {self.max_daily_transactions} transactions per day are suspicious',
            'location_check': 'Transactions from certain countries are considered higher risk'
        }

# Sample usage
if __name__ == "__main__":
    detector = SimpleFraudDetector()
    
    # Test with a sample transaction
    sample_transaction = {
        'amount': 1200.0,
        'hour': 23,
        'n_transactions_day': 6,
        'country': 'RU'
    }
    
    result = detector.predict(sample_transaction)
    print("Prediction Result:", result)
    print("\nRules Description:", detector.get_rules_description())
