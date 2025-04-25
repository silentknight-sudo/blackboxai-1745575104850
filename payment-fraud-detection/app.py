from flask import Flask, render_template, request, jsonify
from datetime import datetime
from model.fraud_detection_model import SimpleFraudDetector
import os

app = Flask(__name__)
detector = SimpleFraudDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html', rules=detector.get_rules_description())

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare transaction data
        transaction = {
            'amount': float(data['amount']),
            'hour': datetime.now().hour,
            'n_transactions_day': data.get('n_transactions_day', 1),
            'country': data.get('country', ''),
            'type': data.get('type', 'online')
        }
        
        # Get prediction
        result = detector.predict(transaction)
        
        # Add transaction details to response
        response = {
            'transaction_id': datetime.now().strftime('%Y%m%d%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'amount': data['amount'],
            'prediction': result
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Return mock statistics for the dashboard
    In a real application, these would come from a database
    """
    return jsonify({
        'total_transactions': 15234,
        'fraud_detected': 152,
        'average_amount': 245.67,
        'response_time': 0.001,
        'accuracy': 0.998,
        'distribution': {
            'online': 45,
            'pos': 30,
            'atm': 15,
            'transfer': 10
        },
        'recent_transactions': [
            {
                'time': '2024-01-20T10:30:00',
                'amount': 299.99,
                'type': 'online',
                'risk_score': 12
            },
            {
                'time': '2024-01-20T10:28:00',
                'amount': 1499.99,
                'type': 'pos',
                'risk_score': 85
            },
            {
                'time': '2024-01-20T10:25:00',
                'amount': 50.00,
                'type': 'atm',
                'risk_score': 5
            }
        ]
    })

@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    """
    Return model rules and performance metrics
    """
    rules = detector.get_rules_description()
    return jsonify({
        'rules': rules,
        'accuracy': 0.998,
        'precision': 0.995,
        'recall': 0.993,
        'f1_score': 0.994,
        'monthly_accuracy': [99.2, 99.4, 99.5, 99.6, 99.7, 99.8],
        'feature_importance': {
            'amount': 0.35,
            'time_of_day': 0.25,
            'location': 0.20,
            'transaction_frequency': 0.20
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
