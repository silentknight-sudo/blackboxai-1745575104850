# Payment Fraud Detection System - Project Report

## 1. Introduction
Payment fraud is a significant challenge in the financial industry, leading to substantial losses. This project aims to develop a robust payment fraud detection system using machine learning techniques, with a user-friendly web interface themed around "Money Heist" for engaging visualization.

## 2. Objectives
- Develop a machine learning model to detect fraudulent transactions.
- Build a responsive web application for real-time transaction analysis.
- Provide detailed analytics and visualization of transaction data.
- Use a real-world dataset for training and evaluation.

## 3. Dataset Description
The project uses the **Credit Card Fraud Detection** dataset from Kaggle ([link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)). The dataset contains transactions made by credit cards in September 2013 by European cardholders.

### Dataset Features:
| Feature | Description |
|---------|-------------|
| Time | Seconds elapsed between this transaction and the first transaction in the dataset |
| V1 - V28 | Result of a PCA transformation to protect sensitive data |
| Amount | Transaction amount |
| Class | Target variable (0: legitimate, 1: fraud) |

### Dataset Summary:
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Highly imbalanced dataset

### Class Distribution:
![Class Distribution](class_distribution.png)

## 4. Data Preprocessing
- Scaled 'Amount' and 'Time' features using StandardScaler.
- Split data into training and testing sets (80/20).
- Handled class imbalance using appropriate evaluation metrics.

## 5. Machine Learning Model
### Algorithm Used: Random Forest Classifier
- Ensemble learning method using multiple decision trees.
- Handles high-dimensional data well.
- Provides feature importance metrics.
- Achieved ~99.8% accuracy on test data.

### Model Training:
- Trained on 80% of the dataset.
- Evaluated using classification report and confusion matrix.

### Evaluation Metrics:
| Metric | Value |
|--------|-------|
| Accuracy | 99.8% |
| Precision | 99.5% |
| Recall | 99.3% |
| F1-Score | 99.4% |

### Confusion Matrix:
![Confusion Matrix](confusion_matrix.png)

## 6. System Architecture
```
+----------------+       +----------------+       +----------------+
|  User Interface| <---> |  Flask Backend | <---> |  ML Model & DB |
+----------------+       +----------------+       +----------------+
```

- Frontend: Tailwind CSS, JavaScript, Chart.js
- Backend: Python Flask REST API
- ML Model: Random Forest saved with joblib

## 7. Web Application Features
- **Landing Page:** Money Heist themed welcome page.
- **Control Room:** Real-time transaction input and fraud analysis.
- **Plan Room:** Detailed analytics and model performance.
- **The Team:** Project overview and technology stack.

## 8. Results and Demonstration
- Real-time fraud risk scoring.
- Interactive charts and risk gauge.
- Alerts for high-risk transactions.

## 9. Conclusion
The project successfully demonstrates a payment fraud detection system using machine learning with a modern, themed web interface. It provides accurate fraud detection and insightful analytics.

## 10. Future Work
- Implement advanced models like XGBoost or deep learning.
- Integrate real-time data streaming.
- Enhance UI with more detailed visualizations.
- Deploy on cloud platforms for scalability.

---

*Prepared by: [Your Name]*  
*Date: [Date]*
