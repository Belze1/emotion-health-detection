/**
 * Xử lý biểu đồ phân tích cảm xúc và đánh giá sức khỏe tinh thần
 */

// Biến lưu trữ các biểu đồ
let emotionTrendChart = null;
let riskScoreChart = null;

// Cấu hình chung cho biểu đồ
const chartConfig = {
    responsive: true,
    maintainAspectRatio: false,  // Cho phép điều chỉnh kích thước
    animation: {
        duration: 1000,
        easing: 'easeInOutQuart'
    },
    plugins: {
        legend: {
            position: 'bottom',
            labels: {
                boxWidth: 20,
                padding: 20
            }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            padding: 10,
            callbacks: {
                label: function(context) {
                    let value = context.raw;
                    return `${context.dataset.label}: ${Math.round(value * 100)}%`;
                }
            }
        }
    },
    scales: {
        y: {
            beginAtZero: true,
            max: 1,
            ticks: {
                callback: value => `${Math.round(value * 100)}%`,
                stepSize: 0.2
            },
            grid: {
                color: 'rgba(0, 0, 0, 0.1)',
                drawBorder: false
            }
        },
        x: {
            grid: {
                display: false
            },
            ticks: {
                maxRotation: 45,
                minRotation: 45
            }
        }
    }
};

// Màu sắc cho các loại cảm xúc
const emotionColors = {
    happy: 'rgba(75, 192, 192, 0.8)',
    sad: 'rgba(54, 162, 235, 0.8)',
    angry: 'rgba(255, 99, 132, 0.8)',
    surprise: 'rgba(255, 206, 86, 0.8)',
    neutral: 'rgba(153, 102, 255, 0.8)'
};

// Màu sắc cho mức độ nguy cơ
const riskColors = {
    low: '#4caf50',
    medium: '#ff9800',
    high: '#f44336'
};

/**
 * Cập nhật biểu đồ xu hướng cảm xúc
 */
function updateEmotionTrendChart(data) {
    const ctx = document.getElementById('emotionTrendChart')?.getContext('2d');
    if (!ctx) return;
    
    if (emotionTrendChart) {
        emotionTrendChart.destroy();
    }

    // Format nhãn thời gian
    const labels = data.labels.map(timestamp => {
        const date = new Date(timestamp);
        return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
    });

    emotionTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Cảm xúc tiêu cực',
                    data: data.negative_ratios,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2
                },
                {
                    label: 'Độ biến thiên',
                    data: data.variances,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2
                }
            ]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: {
                    display: true,
                    text: 'Xu Hướng Cảm Xúc Theo Thời Gian',
                    padding: 20
                }
            }
        }
    });
}

/**
 * Cập nhật biểu đồ điểm đánh giá nguy cơ
 */
function updateRiskScoreChart(data) {
    const ctx = document.getElementById('riskScoreChart')?.getContext('2d');
    if (!ctx) return;
    
    if (riskScoreChart) {
        riskScoreChart.destroy();
    }

    // Format nhãn thời gian
    const labels = data.labels.map(timestamp => {
        const date = new Date(timestamp);
        return `${date.getDate()}/${date.getMonth() + 1} ${date.getHours()}:${date.getMinutes()}`;
    });

    riskScoreChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Điểm nguy cơ',
                data: data.risk_scores,
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointBackgroundColor: data.risk_scores.map(score => {
                    if (score >= 0.7) return riskColors.high;
                    if (score >= 0.4) return riskColors.medium;
                    return riskColors.low;
                }),
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: {
                    display: true,
                    text: 'Điểm Đánh Giá Nguy Cơ Theo Thời Gian',
                    padding: 20
                }
            }
        }
    });
}

/**
 * Cập nhật hiển thị các chỉ số
 */
function updateMetrics(data) {
    // Cảm xúc
    if (data.emotion_metrics) {
        document.getElementById('negativeRatio').textContent = 
            `${Math.round(data.emotion_metrics.negative_ratio * 100)}%`;
        document.getElementById('emotionVariance').textContent = 
            data.emotion_metrics.emotion_variance.toFixed(3);
    }

    // Thời gian sử dụng
    if (data.usage_metrics) {
        document.getElementById('continuousUsage').textContent = 
            `${data.usage_metrics.max_continuous_usage.toFixed(1)} giờ`;
        document.getElementById('lateNightUsage').textContent = 
            `${Math.round(data.usage_metrics.late_night_ratio * 100)}%`;
    }

    // Tương tác
    if (data.interaction_metrics) {
        document.getElementById('dislikeRatio').textContent = 
            `${Math.round(data.interaction_metrics.dislike_ratio * 100)}%`;
        document.getElementById('totalInteractions').textContent = 
            data.interaction_metrics.total_interactions;
    }
}

/**
 * Cập nhật kết quả đánh giá
 */
function updateAssessmentResult(data) {
    const resultDiv = document.getElementById('assessmentResult');
    if (!resultDiv) return;

    let riskClass = 'success';
    let riskText = 'Thấp';
    
    if (data.risk_level === 'medium') {
        riskClass = 'warning';
        riskText = 'Trung bình';
    } else if (data.risk_level === 'high') {
        riskClass = 'danger';
        riskText = 'Cao';
    }
    
    resultDiv.innerHTML = `
        <div class="alert alert-${riskClass} assessment-result risk-${data.risk_level}">
            <h6>Mức độ nguy cơ: ${riskText}</h6>
            <p>Điểm đánh giá: ${Math.round(data.risk_score * 100)}%</p>
            <p>Thời gian: ${new Date(data.timestamp).toLocaleString()}</p>
        </div>
    `;

    // Cập nhật style cho kết quả
    const resultBox = resultDiv.querySelector('.assessment-result');
    resultBox.classList.add('fade-in');
}