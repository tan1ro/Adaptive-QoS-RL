// Dashboard JavaScript for Adaptive QoS RL

const API_BASE = 'http://localhost:8080/api/v1';
const UPDATE_INTERVAL = 1000; // Update every second

// Chart instances
let rewardChart = null;
let networkChart = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    startUpdates();
    checkHealth();
});

// Initialize Chart.js charts
function initializeCharts() {
    // Reward Chart
    const rewardCtx = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(rewardCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Episode Reward',
                data: [],
                borderColor: 'rgb(37, 99, 235)',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Average Reward',
                data: [],
                borderColor: 'rgb(124, 58, 237)',
                backgroundColor: 'rgba(124, 58, 237, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Network State Chart
    const networkCtx = document.getElementById('networkChart').getContext('2d');
    networkChart = new Chart(networkCtx, {
        type: 'bar',
        data: {
            labels: ['Link Utilization', 'Queue Length', 'Delay', 'Packet Loss'],
            datasets: [{
                label: 'Network Metrics',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(37, 99, 235, 0.8)',
                    'rgba(124, 58, 237, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgb(37, 99, 235)',
                    'rgb(124, 58, 237)',
                    'rgb(245, 158, 11)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateStatus(true);
        } else {
            updateStatus(false);
        }
    } catch (error) {
        updateStatus(false);
        console.error('Health check failed:', error);
    }
}

// Update connection status
function updateStatus(connected) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    if (connected) {
        statusDot.classList.add('connected');
        statusText.textContent = 'Connected';
        statusText.style.color = '#10b981';
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
        statusText.style.color = '#ef4444';
    }
}

// Start periodic updates
function startUpdates() {
    updateAll();
    setInterval(updateAll, UPDATE_INTERVAL);
}

// Update all dashboard data
async function updateAll() {
    await updateTrainingMetrics();
    await updateNetworkState();
    await updateFlowStats();
}

// Update training metrics
async function updateTrainingMetrics() {
    try {
        const response = await fetch(`${API_BASE}/training/metrics`);
        const result = await response.json();
        
        if (result.success) {
            const metrics = result.metrics;
            
            // Update UI elements
            document.getElementById('currentEpisode').textContent = metrics.episode || 0;
            document.getElementById('totalEpisodes').textContent = metrics.total_episodes || 0;
            document.getElementById('currentReward').textContent = (metrics.current_reward || 0).toFixed(2);
            document.getElementById('averageReward').textContent = (metrics.average_reward || 0).toFixed(2);
            document.getElementById('epsilon').textContent = (metrics.epsilon || 1.0).toFixed(3);
            document.getElementById('loss').textContent = (metrics.loss || 0).toFixed(3);
            
            // Update progress bars
            const episodeProgress = metrics.total_episodes > 0 
                ? (metrics.episode / metrics.total_episodes) * 100 
                : 0;
            document.getElementById('episodeProgress').style.width = `${episodeProgress}%`;
            document.getElementById('epsilonProgress').style.width = `${(metrics.epsilon || 1.0) * 100}%`;
            
            // Update status
            const statusEl = document.getElementById('trainingStatus');
            if (metrics.is_training) {
                statusEl.textContent = 'Training';
                statusEl.className = 'metric-value status-training';
            } else if (metrics.episode > 0 && metrics.episode >= metrics.total_episodes) {
                statusEl.textContent = 'Complete';
                statusEl.className = 'metric-value status-complete';
            } else {
                statusEl.textContent = 'Idle';
                statusEl.className = 'metric-value status-idle';
            }
            
            // Update reward chart
            if (metrics.scores && metrics.scores.length > 0) {
                const scores = metrics.scores;
                const labels = scores.map((_, i) => `E${i + 1}`);
                const avgScores = [];
                
                for (let i = 0; i < scores.length; i++) {
                    const window = scores.slice(Math.max(0, i - 9), i + 1);
                    avgScores.push(window.reduce((a, b) => a + b, 0) / window.length);
                }
                
                rewardChart.data.labels = labels.slice(-50); // Last 50 points
                rewardChart.data.datasets[0].data = scores.slice(-50);
                rewardChart.data.datasets[1].data = avgScores.slice(-50);
                rewardChart.update('none');
            }
        }
    } catch (error) {
        console.error('Failed to update training metrics:', error);
    }
}

// Update network state
async function updateNetworkState() {
    try {
        const response = await fetch(`${API_BASE}/state`);
        const result = await response.json();
        
        if (result.success && result.state) {
            const state = result.state;
            
            // Update network metrics display
            updateMetricDisplay('linkUtil', state.link_utilization || [0]);
            updateMetricDisplay('queueLength', state.queue_length || [0]);
            updateMetricDisplay('delay', state.delay || [0]);
            updateMetricDisplay('packetLoss', state.packet_loss || [0]);
            
            // Update network chart
            const avgUtil = calculateAverage(state.link_utilization || [0]);
            const avgQueue = calculateAverage(state.queue_length || [0]);
            const avgDelay = calculateAverage(state.delay || [0]);
            const avgLoss = calculateAverage(state.packet_loss || [0]);
            
            networkChart.data.datasets[0].data = [
                Math.min(avgUtil, 1.0),
                Math.min(avgQueue, 1.0),
                Math.min(avgDelay, 1.0),
                Math.min(avgLoss, 1.0)
            ];
            networkChart.update('none');
        }
    } catch (error) {
        console.error('Failed to update network state:', error);
    }
}

// Update flow statistics table
async function updateFlowStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const result = await response.json();
        
        if (result.success && result.stats) {
            const stats = result.stats;
            const tbody = document.getElementById('statsTableBody');
            
            if (Object.keys(stats).length === 0) {
                tbody.innerHTML = '<tr><td colspan="5">No statistics available</td></tr>';
                return;
            }
            
            let html = '';
            for (const [dpid, flows] of Object.entries(stats)) {
                if (typeof flows === 'object' && flows !== null) {
                    for (const [port, flowStats] of Object.entries(flows)) {
                        html += `
                            <tr>
                                <td>${dpid}</td>
                                <td>${port}</td>
                                <td>${flowStats.packet_count || 0}</td>
                                <td>${formatBytes(flowStats.byte_count || 0)}</td>
                                <td>${formatDuration(flowStats.duration_sec || 0, flowStats.duration_nsec || 0)}</td>
                            </tr>
                        `;
                    }
                }
            }
            
            tbody.innerHTML = html || '<tr><td colspan="5">No statistics available</td></tr>';
        }
    } catch (error) {
        console.error('Failed to update flow stats:', error);
    }
}

// Helper functions
function updateMetricDisplay(elementId, values) {
    const container = document.getElementById(elementId);
    if (!container) return;
    
    if (!Array.isArray(values) || values.length === 0) {
        container.innerHTML = '<div class="metric-cell">N/A</div>';
        return;
    }
    
    container.innerHTML = values.map((val, idx) => 
        `<div class="metric-cell">Port ${idx}: ${(val * 100).toFixed(1)}%</div>`
    ).join('');
}

function calculateAverage(values) {
    if (!Array.isArray(values) || values.length === 0) return 0;
    const sum = values.reduce((a, b) => a + b, 0);
    return sum / values.length;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(sec, nsec) {
    const total = sec + (nsec / 1e9);
    if (total < 1) return `${(total * 1000).toFixed(0)}ms`;
    if (total < 60) return `${total.toFixed(2)}s`;
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${minutes}m ${seconds.toFixed(0)}s`;
}

