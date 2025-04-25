// Utility function for API calls
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Dashboard functionality
if (document.getElementById('transaction-form')) {
    const form = document.getElementById('transaction-form');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const amount = document.getElementById('amount').value;
        const cardNumber = document.getElementById('card_number').value;
        const type = document.getElementById('type').value;

        try {
            const result = await fetchAPI('/api/predict', {
                method: 'POST',
                body: JSON.stringify({
                    amount: parseFloat(amount),
                    card_number: cardNumber,
                    type: type
                })
            });

            // Update risk gauge
            updateRiskGauge(result.prediction.risk_score);

            // Add transaction to table
            addTransactionToTable({
                time: new Date().toLocaleTimeString(),
                amount: amount,
                type: type,
                risk: result.prediction.risk_score
            });

            // Show alert based on risk score
            showAlert(result.prediction.risk_score);

        } catch (error) {
            showAlert('Error processing transaction', 'error');
        }
    });
}

// Update risk gauge
function updateRiskGauge(score) {
    const gauge = Chart.getChart('riskGauge');
    if (gauge) {
        gauge.data.datasets[0].data = [score, 100 - score];
        gauge.update();
        document.getElementById('risk-score').textContent = `${Math.round(score)}%`;
    }
}

// Add transaction to table
function addTransactionToTable(transaction) {
    const tbody = document.getElementById('transactions-table');
    if (!tbody) return;

    const row = tbody.insertRow(0);
    const riskClass = transaction.risk > 70 ? 'text-red-600' : 
                     transaction.risk > 30 ? 'text-yellow-600' : 
                     'text-green-600';

    row.innerHTML = `
        <td class="py-2">${transaction.time}</td>
        <td class="py-2">$${transaction.amount}</td>
        <td class="py-2">${transaction.type}</td>
        <td class="py-2 ${riskClass}">${Math.round(transaction.risk)}%</td>
    `;
}

// Show alert
function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `fixed top-4 right-4 p-4 rounded-lg ${
        type === 'error' ? 'bg-red-600' : 'bg-green-600'
    } text-white`;
    alertDiv.textContent = typeof message === 'string' ? message : 
        `Risk Score: ${Math.round(message)}%`;

    document.body.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 3000);
}

// Analysis page functionality
if (document.getElementById('modelPerformanceChart')) {
    fetchAPI('/api/model-performance')
        .then(data => {
            updateModelPerformanceChart(data);
            updateFeatureImportance(data.feature_importance);
        })
        .catch(error => console.error('Error loading model performance:', error));
}

// Update model performance chart
function updateModelPerformanceChart(data) {
    const ctx = document.getElementById('modelPerformanceChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Model Accuracy',
                data: data.monthly_accuracy,
                borderColor: '#FF0000',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 98,
                    max: 100
                }
            }
        }
    });
}

// Update feature importance
function updateFeatureImportance(data) {
    const container = document.querySelector('.feature-importance');
    if (!container) return;

    Object.entries(data).forEach(([feature, importance]) => {
        const div = document.createElement('div');
        div.className = 'mb-4';
        div.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <span>${feature}</span>
                <span class="text-red-600">${(importance * 100).toFixed(1)}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-red-600 h-2 rounded-full" 
                     style="width: ${importance * 100}%"></div>
            </div>
        `;
        container.appendChild(div);
    });
}

// Initialize dashboard charts
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('fraudChart')) {
        initializeDashboardCharts();
    }
});

// Initialize dashboard charts
function initializeDashboardCharts() {
    fetchAPI('/api/stats')
        .then(data => {
            updateTransactionStats(data);
            initializeCharts(data);
        })
        .catch(error => console.error('Error loading dashboard stats:', error));
}

// Update transaction statistics
function updateTransactionStats(data) {
    const stats = {
        'total-transactions': data.total_transactions,
        'fraud-detected': data.fraud_detected,
        'average-amount': `$${data.average_amount}`,
        'response-time': `${data.response_time}s`
    };

    Object.entries(stats).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });
}

// Initialize charts
function initializeCharts(data) {
    // Fraud trend chart
    const fraudCtx = document.getElementById('fraudChart').getContext('2d');
    new Chart(fraudCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 10}, (_, i) => `${i}m ago`),
            datasets: [{
                label: 'Fraud Risk Score',
                data: Array.from({length: 10}, () => Math.random() * 100),
                borderColor: '#FF0000',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

    // Initialize risk gauge
    const gaugeCtx = document.getElementById('riskGauge').getContext('2d');
    new Chart(gaugeCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: ['#FF0000', '#333333'],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '80%',
            rotation: -90,
            circumference: 180,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}
