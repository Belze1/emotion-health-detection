// Emotion Chart Configuration
const emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral'];
const emotionColors = {
    happy: 'rgb(75, 192, 192)',
    sad: 'rgb(54, 162, 235)',
    angry: 'rgb(255, 99, 132)',
    surprise: 'rgb(255, 206, 86)',
    neutral: 'rgb(153, 102, 255)'
};

// Initialize real-time emotion chart
function initEmotionChart() {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: emotions.map(emotion => ({
                label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                data: [],
                borderColor: emotionColors[emotion],
                fill: false,
                tension: 0.4
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                    display: true,
                    text: 'Probability'
                    }
                },
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            },
            animation: {
                duration: 500
            }
        }
    });
}

// Initialize risk timeline chart
function initRiskChart() {
    const ctx = document.getElementById('riskTimeline').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Depression Risk',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Risk Level'
                    }
                },
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            },
            animation: {
                duration: 500
            }
        }
    });
}

// Update emotion chart with new data
function updateEmotionChart(chart, emotionData) {
    const now = new Date().toLocaleTimeString();
    
    // Add new data point
    chart.data.labels.push(now);
    emotions.forEach((emotion, index) => {
        chart.data.datasets[index].data.push(emotionData[emotion]);
    });
    
    // Remove old data points if more than 20
    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => dataset.data.shift());
    }
    
    chart.update('none'); // Update without animation for smooth real-time updates
}

// Update risk timeline with new data
function updateRiskChart(chart, riskScore) {
    const now = new Date().toLocaleTimeString();
    
    // Add new data point
    chart.data.labels.push(now);
    chart.data.datasets[0].data.push(riskScore);
    
    // Remove old data points if more than 50
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.update('none');
}

// Format timestamps for display
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Create emotion history visualization
function createEmotionHistory(data, containerId) {
    const container = document.getElementById(containerId);
    const width = container.offsetWidth;
    const height = 200;
    
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
        
    // Create stacked area chart
    const stack = d3.stack()
        .keys(emotions)
        .order(d3.stackOrderNone)
        .offset(d3.stackOffsetNone);
        
    const series = stack(data);
    
    // Create scales
    const x = d3.scaleTime()
        .domain(d3.extent(data, d => new Date(d.timestamp)))
        .range([0, width]);
        
    const y = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);
        
    // Create area generator
    const area = d3.area()
        .x(d => x(new Date(d.data.timestamp)))
        .y0(d => y(d[0]))
        .y1(d => y(d[1]))
        .curve(d3.curveBasis);
        
    // Add areas
    svg.selectAll('path')
        .data(series)
        .enter()
        .append('path')
        .attr('d', area)
        .style('fill', (d, i) => emotionColors[emotions[i]])
        .style('opacity', 0.7);
}