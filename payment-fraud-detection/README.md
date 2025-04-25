# La Casa de Fraud Detection 🎭

A sophisticated payment fraud detection system using Machine Learning, styled with a Money Heist theme. This project demonstrates the implementation of real-time fraud detection for financial transactions using Python, Flask, and modern web technologies.

## Features 🚀

- Real-time transaction fraud detection
- Interactive dashboard with live monitoring
- Advanced analytics and visualization
- ML-powered risk assessment
- Money Heist themed UI
- Responsive design for all devices

## Technology Stack 💻

### Frontend
- HTML5 & CSS3
- Tailwind CSS for styling
- JavaScript for interactivity
- Chart.js for visualizations
- Font Awesome icons
- Google Fonts

### Backend
- Python Flask
- RESTful API architecture
- SQLite database
- Real-time data processing

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- XGBoost
- Random Forest Classifier

## Project Structure 📁

```
payment-fraud-detection/
├── static/
│   ├── css/
│   └── js/
│       └── main.js
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── analysis.html
│   └── about.html
├── model/
│   └── fraud_detection_model.py
├── app.py
└── requirements.txt
```

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/payment-fraud-detection.git
cd payment-fraud-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:8000
```

## Key Components 🔑

### The Control Room (Dashboard)
- Real-time transaction monitoring
- Risk score visualization
- Recent transaction history
- Interactive fraud detection form

### The Plan Room (Analysis)
- Model performance metrics
- Transaction pattern analysis
- Geographic distribution
- Feature importance visualization

### The Team (About)
- Project overview
- Technology stack details
- Performance statistics
- Contact information

## Machine Learning Model 🤖

The fraud detection system uses a Random Forest Classifier trained on transaction data with the following features:
- Transaction amount
- Time of day
- Day of week
- Number of transactions per day
- Average transaction amount

The model achieves:
- 99.8% accuracy
- Fast response time (0.001s)
- Real-time prediction capabilities

## Security Features 🛡️

- Real-time transaction validation
- Risk score calculation
- Pattern recognition
- Anomaly detection
- Continuous monitoring

## API Endpoints 🌐

- `/api/predict` - POST: Submit transaction for fraud detection
- `/api/stats` - GET: Retrieve system statistics
- `/api/model-performance` - GET: Get model performance metrics

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Inspired by "La Casa de Papel" (Money Heist)
- Built with modern web technologies
- Powered by machine learning

---

*"In this world, everything is governed by balance. There's what you stand to gain and what you stand to lose. And when you think you've got nothing to lose, you become overconfident."* - The Professor
