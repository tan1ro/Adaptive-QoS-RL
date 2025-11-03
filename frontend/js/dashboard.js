/**
 * Adaptive QoS RL Dashboard - JavaScript
 * 
 * This file handles all frontend functionality:
 * - Real-time data fetching from REST API
 * - Chart.js visualization initialization and updates
 * - UI element updates for training metrics
 * - Network state visualization
 * - Flow statistics table management
 * - Connection status monitoring
 */

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Base URL for all API endpoints
// All API calls will be prefixed with this URL
const API_BASE = 'http://localhost:8080/api/v1';

// Update interval in milliseconds
// Dashboard will refresh data every second (1000ms)
// Lower values = more frequent updates but higher server load
const UPDATE_INTERVAL = 1000;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// Chart.js chart instances (initialized later)
// These are global so they can be accessed and updated from any function
let rewardChart = null;      // Line chart showing episode rewards over time
let networkChart = null;     // Bar chart showing network state metrics

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize dashboard when page loads
 * 
 * This event listener waits for the HTML document to be fully loaded
 * before executing initialization functions
 */
document.addEventListener('DOMContentLoaded', () => {
    // Step 1: Initialize Chart.js charts
    // Sets up the canvas elements and chart configurations
    initializeCharts();
    
    // Step 2: Start periodic updates
    // Begins fetching and updating data every UPDATE_INTERVAL milliseconds
    startUpdates();
    
    // Step 3: Check API health
    // Verifies the backend is running and accessible
    checkHealth();
});

// ============================================================================
// CHART INITIALIZATION
// ============================================================================

/**
 * Initialize Chart.js charts for data visualization
 * 
 * Creates two charts:
 * 1. Reward Chart: Line chart showing episode rewards and averages
 * 2. Network Chart: Bar chart showing network state metrics
 */
function initializeCharts() {
    // ========================================================================
    // REWARD CHART - Shows training progress over episodes
    // ========================================================================
    
    // Get the canvas element from HTML by its ID
    // getContext('2d') gets the 2D rendering context needed for drawing
    const rewardCtx = document.getElementById('rewardChart').getContext('2d');
    
    // Create new Chart.js line chart instance
    rewardChart = new Chart(rewardCtx, {
        type: 'line',  // Line chart type for time series data
        
        // Chart data configuration
        data: {
            labels: [],  // X-axis labels (episode numbers) - empty initially
            
            // Data series to plot
            datasets: [
                {
                    label: 'Episode Reward',        // Legend label for this dataset
                    data: [],                       // Y-axis values (rewards) - empty initially
                    borderColor: 'rgb(37, 99, 235)', // Line color (blue)
                    backgroundColor: 'rgba(37, 99, 235, 0.1)', // Fill color (light blue, 10% opacity)
                    tension: 0.4,                   // Bezier curve tension (smoothness)
                    fill: true                      // Fill area under the line
                },
                {
                    label: 'Average Reward',        // Second dataset for moving average
                    data: [],                       // Average values - empty initially
                    borderColor: 'rgb(124, 58, 237)', // Line color (purple)
                    backgroundColor: 'rgba(124, 58, 237, 0.1)', // Fill color (light purple)
                    tension: 0.4,                   // Smooth curve
                    fill: true                      // Fill area
                }
            ]
        },
        
        // Chart display options
        options: {
            responsive: true,              // Chart resizes with container
            maintainAspectRatio: false,     // Allow custom height (set by CSS)
            
            // Plugin configurations
            plugins: {
                legend: {
                    display: true,         // Show legend
                    position: 'top'        // Position legend at top
                },
                tooltip: {
                    mode: 'index',         // Show all datasets at same X value
                    intersect: false       // Tooltip appears when hovering near point
                }
            },
            
            // Axis configurations
            scales: {
                y: {
                    beginAtZero: true,     // Y-axis starts at 0
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)' // Light gray grid lines
                    }
                },
                x: {
                    grid: {
                        display: false    // Hide vertical grid lines for cleaner look
                    }
                }
            }
        }
    });

    // ========================================================================
    // NETWORK STATE CHART - Shows current network metrics
    // ========================================================================
    
    // Get canvas element for network chart
    const networkCtx = document.getElementById('networkChart').getContext('2d');
    
    // Create new Chart.js bar chart instance
    networkChart = new Chart(networkCtx, {
        type: 'bar',  // Bar chart type for categorical data
        
        // Chart data configuration
        data: {
            // X-axis labels (metric names)
            labels: ['Link Utilization', 'Queue Length', 'Delay', 'Packet Loss'],
            
            // Data series
            datasets: [{
                label: 'Network Metrics',  // Legend label
                data: [0, 0, 0, 0],        // Initial values (all zero)
                
                // Colors for each bar (in order)
                backgroundColor: [
                    'rgba(37, 99, 235, 0.8)',  // Blue for utilization
                    'rgba(124, 58, 237, 0.8)', // Purple for queue
                    'rgba(245, 158, 11, 0.8)',  // Orange for delay
                    'rgba(239, 68, 68, 0.8)'    // Red for packet loss
                ],
                
                // Border colors for each bar
                borderColor: [
                    'rgb(37, 99, 235)',
                    'rgb(124, 58, 237)',
                    'rgb(245, 158, 11)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 2  // Bar border width in pixels
            }]
        },
        
        // Chart display options
        options: {
            responsive: true,              // Responsive sizing
            maintainAspectRatio: false,    // Custom height
            
            // Plugin configurations
            plugins: {
                legend: {
                    display: false         // Hide legend (only one dataset)
                }
            },
            
            // Axis configurations
            scales: {
                y: {
                    beginAtZero: true,     // Y-axis starts at 0
                    max: 1,                // Maximum value (metrics normalized 0-1)
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)' // Light gray grid
                    }
                }
            }
        }
    });
}

// ============================================================================
// HEALTH CHECK & CONNECTION STATUS
// ============================================================================

/**
 * Check if the backend API is accessible
 * 
 * Sends a GET request to the health endpoint to verify:
 * - Backend server is running
 * - Network connectivity is working
 * - API endpoints are responding
 */
async function checkHealth() {
    try {
        // Send GET request to health endpoint
        // fetch() is a modern JavaScript API for HTTP requests
        const response = await fetch(`${API_BASE}/health`);
        
        // Parse JSON response from server
        // Response format: {"status": "healthy", "controller": "running"}
        const data = await response.json();
        
        // Check if status indicates healthy system
        if (data.status === 'healthy') {
            // Update UI to show connected status
            updateStatus(true);
        } else {
            // Update UI to show disconnected status
            updateStatus(false);
        }
    } catch (error) {
        // If request fails (network error, server down, etc.)
        // Update UI to show disconnected and log error for debugging
        updateStatus(false);
        console.error('Health check failed:', error);
    }
}

/**
 * Update connection status indicator in the UI
 * 
 * Changes the status dot color and text based on connection state
 * 
 * @param {boolean} connected - True if connected, false if disconnected
 */
function updateStatus(connected) {
    // Get DOM elements by their IDs
    const statusDot = document.getElementById('statusDot');   // Colored circle
    const statusText = document.getElementById('statusText'); // Status text
    
    if (connected) {
        // Add 'connected' CSS class (makes dot green)
        statusDot.classList.add('connected');
        
        // Update status text
        statusText.textContent = 'Connected';
        
        // Set text color to green
        statusText.style.color = '#10b981';
    } else {
        // Remove 'connected' class (makes dot red)
        statusDot.classList.remove('connected');
        
        // Update status text
        statusText.textContent = 'Disconnected';
        
        // Set text color to red
        statusText.style.color = '#ef4444';
    }
}

// ============================================================================
// PERIODIC UPDATES
// ============================================================================

/**
 * Start periodic updates of all dashboard data
 * 
 * Calls updateAll() immediately, then every UPDATE_INTERVAL milliseconds
 */
function startUpdates() {
    // Update immediately when called (don't wait for first interval)
    updateAll();
    
    // Set up interval to call updateAll() repeatedly
    // setInterval() schedules a function to run periodically
    setInterval(updateAll, UPDATE_INTERVAL);
}

/**
 * Update all dashboard sections with latest data
 * 
 * This is the main update function called periodically
 * It fetches data for all sections in parallel (async/await)
 */
async function updateAll() {
    // Update training metrics (episode, reward, epsilon, loss, etc.)
    await updateTrainingMetrics();
    
    // Update network state (utilization, queue, delay, packet loss)
    await updateNetworkState();
    
    // Update flow statistics table
    await updateFlowStats();
}

// ============================================================================
// TRAINING METRICS UPDATES
// ============================================================================

/**
 * Fetch and update training metrics from the API
 * 
 * This function:
 * 1. Fetches current training metrics from REST API
 * 2. Updates all UI elements showing training progress
 * 3. Updates the reward chart with new data points
 */
async function updateTrainingMetrics() {
    try {
        // Send GET request to training metrics endpoint
        const response = await fetch(`${API_BASE}/training/metrics`);
        
        // Parse JSON response
        // Expected format: {"success": true, "metrics": {...}}
        const result = await response.json();
        
        // Check if request was successful
        if (result.success) {
            // Extract metrics object from response
            const metrics = result.metrics;
            
            // ================================================================
            // UPDATE UI ELEMENTS WITH CURRENT VALUES
            // ================================================================
            
            // Update episode number (current episode / total episodes)
            // || 0 provides default value if undefined
            document.getElementById('currentEpisode').textContent = metrics.episode || 0;
            document.getElementById('totalEpisodes').textContent = metrics.total_episodes || 0;
            
            // Update reward displays
            // toFixed(2) formats number to 2 decimal places
            document.getElementById('currentReward').textContent = (metrics.current_reward || 0).toFixed(2);
            document.getElementById('averageReward').textContent = (metrics.average_reward || 0).toFixed(2);
            
            // Update exploration rate (epsilon)
            // Epsilon decreases as agent learns (less exploration, more exploitation)
            document.getElementById('epsilon').textContent = (metrics.epsilon || 1.0).toFixed(3);
            
            // Update training loss
            // Lower loss = better learning progress
            document.getElementById('loss').textContent = (metrics.loss || 0).toFixed(3);
            
            // ================================================================
            // UPDATE PROGRESS BARS
            // ================================================================
            
            // Calculate episode progress percentage
            // Formula: (current_episode / total_episodes) * 100
            const episodeProgress = metrics.total_episodes > 0 
                ? (metrics.episode / metrics.total_episodes) * 100 
                : 0;
            
            // Update episode progress bar width
            // CSS style.width controls visual progress
            document.getElementById('episodeProgress').style.width = `${episodeProgress}%`;
            
            // Update epsilon progress bar
            // Epsilon goes from 1.0 (100% exploration) to 0.0 (0% exploration)
            document.getElementById('epsilonProgress').style.width = `${(metrics.epsilon || 1.0) * 100}%`;
            
            // ================================================================
            // UPDATE TRAINING STATUS
            // ================================================================
            
            // Get status element from DOM
            const statusEl = document.getElementById('trainingStatus');
            
            // Determine status based on training state and progress
            if (metrics.is_training) {
                // Currently training - show active status
                statusEl.textContent = 'Training';
                statusEl.className = 'metric-value status-training'; // Apply CSS class
            } else if (metrics.episode > 0 && metrics.episode >= metrics.total_episodes) {
                // Training completed - all episodes finished
                statusEl.textContent = 'Complete';
                statusEl.className = 'metric-value status-complete'; // Green color
            } else {
                // Not training - idle state
                statusEl.textContent = 'Idle';
                statusEl.className = 'metric-value status-idle'; // Gray color
            }
            
            // ================================================================
            // UPDATE REWARD CHART
            // ================================================================
            
            // Check if we have score data to plot
            if (metrics.scores && metrics.scores.length > 0) {
                const scores = metrics.scores; // Array of episode rewards
                
                // Create labels for X-axis (E1, E2, E3, ...)
                const labels = scores.map((_, i) => `E${i + 1}`);
                
                // Calculate moving average for each point
                // Moving average smooths out noise and shows trend
                const avgScores = [];
                
                // For each score, calculate average of last 10 episodes
                for (let i = 0; i < scores.length; i++) {
                    // Get window of last 10 episodes (or fewer if not enough data)
                    const window = scores.slice(Math.max(0, i - 9), i + 1);
                    
                    // Calculate average of window
                    const avg = window.reduce((a, b) => a + b, 0) / window.length;
                    avgScores.push(avg);
                }
                
                // Update chart data with last 50 points (to avoid overcrowding)
                // slice(-50) gets last 50 elements from array
                rewardChart.data.labels = labels.slice(-50);
                rewardChart.data.datasets[0].data = scores.slice(-50);
                rewardChart.data.datasets[1].data = avgScores.slice(-50);
                
                // Update chart display (without animation for performance)
                // 'none' means no transition animation
                rewardChart.update('none');
            }
        }
    } catch (error) {
        // Log error but don't crash - dashboard continues working
        console.error('Failed to update training metrics:', error);
    }
}

// ============================================================================
// NETWORK STATE UPDATES
// ============================================================================

/**
 * Fetch and update network state metrics from the API
 * 
 * This function:
 * 1. Fetches current network state (utilization, queue, delay, loss)
 * 2. Updates metric display cells
 * 3. Updates the network state bar chart
 */
async function updateNetworkState() {
    try {
        // Send GET request to state endpoint
        const response = await fetch(`${API_BASE}/state`);
        
        // Parse JSON response
        const result = await response.json();
        
        // Check if request succeeded and state data exists
        if (result.success && result.state) {
            const state = result.state; // Extract state object
            
            // ================================================================
            // UPDATE METRIC DISPLAY CELLS
            // ================================================================
            
            // Update each metric display with array of values
            // Each metric has values per port/switch
            updateMetricDisplay('linkUtil', state.link_utilization || [0]);
            updateMetricDisplay('queueLength', state.queue_length || [0]);
            updateMetricDisplay('delay', state.delay || [0]);
            updateMetricDisplay('packetLoss', state.packet_loss || [0]);
            
            // ================================================================
            // UPDATE NETWORK CHART
            // ================================================================
            
            // Calculate average value for each metric
            // This gives a single representative value for the chart
            const avgUtil = calculateAverage(state.link_utilization || [0]);
            const avgQueue = calculateAverage(state.queue_length || [0]);
            const avgDelay = calculateAverage(state.delay || [0]);
            const avgLoss = calculateAverage(state.packet_loss || [0]);
            
            // Update chart data
            // Math.min() ensures values don't exceed 1.0 (normalized range)
            networkChart.data.datasets[0].data = [
                Math.min(avgUtil, 1.0),   // Link utilization (0-100%)
                Math.min(avgQueue, 1.0),  // Queue length (normalized)
                Math.min(avgDelay, 1.0),  // Delay (normalized)
                Math.min(avgLoss, 1.0)    // Packet loss (normalized)
            ];
            
            // Update chart display (no animation for performance)
            networkChart.update('none');
        }
    } catch (error) {
        // Log error but continue operation
        console.error('Failed to update network state:', error);
    }
}

// ============================================================================
// FLOW STATISTICS UPDATES
// ============================================================================

/**
 * Fetch and update flow statistics table
 * 
 * This function:
 * 1. Fetches flow statistics from all switches
 * 2. Populates the statistics table with switch/port data
 * 3. Formats values for human readability
 */
async function updateFlowStats() {
    try {
        // Send GET request to stats endpoint
        const response = await fetch(`${API_BASE}/stats`);
        
        // Parse JSON response
        const result = await response.json();
        
        // Check if request succeeded and stats exist
        if (result.success && result.stats) {
            const stats = result.stats; // Stats object: {dpid: {port: {stats}}}
            
            // Get table body element where we'll insert rows
            const tbody = document.getElementById('statsTableBody');
            
            // Check if stats object is empty
            if (Object.keys(stats).length === 0) {
                // Show message if no statistics available
                tbody.innerHTML = '<tr><td colspan="5">No statistics available</td></tr>';
                return;
            }
            
            // Build HTML table rows from stats data
            let html = '';
            
            // Iterate over switches (DPID = DataPath ID, unique switch identifier)
            for (const [dpid, flows] of Object.entries(stats)) {
                // Check if flows is a valid object
                if (typeof flows === 'object' && flows !== null) {
                    // Iterate over ports on this switch
                    for (const [port, flowStats] of Object.entries(flows)) {
                        // Build HTML row with statistics
                        html += `
                            <tr>
                                <td>${dpid}</td>                                          <!-- Switch ID -->
                                <td>${port}</td>                                          <!-- Port number -->
                                <td>${flowStats.packet_count || 0}</td>                  <!-- Packet count -->
                                <td>${formatBytes(flowStats.byte_count || 0)}</td>       <!-- Bytes (formatted) -->
                                <td>${formatDuration(flowStats.duration_sec || 0, flowStats.duration_nsec || 0)}</td> <!-- Duration (formatted) -->
                            </tr>
                        `;
                    }
                }
            }
            
            // Insert HTML into table (or show message if no data)
            tbody.innerHTML = html || '<tr><td colspan="5">No statistics available</td></tr>';
        }
    } catch (error) {
        // Log error but continue operation
        console.error('Failed to update flow stats:', error);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Update a metric display container with array of values
 * 
 * Creates metric cells showing values for each port/switch
 * 
 * @param {string} elementId - ID of container element
 * @param {Array<number>} values - Array of metric values (one per port)
 */
function updateMetricDisplay(elementId, values) {
    // Get container element from DOM
    const container = document.getElementById(elementId);
    
    // Return early if element doesn't exist
    if (!container) return;
    
    // Check if values array is valid and non-empty
    if (!Array.isArray(values) || values.length === 0) {
        // Show "N/A" if no data available
        container.innerHTML = '<div class="metric-cell">N/A</div>';
        return;
    }
    
    // Create HTML for each value
    // map() creates new array by transforming each value
    // join('') concatenates array elements into single string
    container.innerHTML = values.map((val, idx) => 
        // Template literal: creates HTML with formatted value
        // (val * 100).toFixed(1) converts normalized value (0-1) to percentage
        `<div class="metric-cell">Port ${idx}: ${(val * 100).toFixed(1)}%</div>`
    ).join('');
}

/**
 * Calculate average of an array of numbers
 * 
 * @param {Array<number>} values - Array of numbers to average
 * @returns {number} Average value (0 if array is empty or invalid)
 */
function calculateAverage(values) {
    // Return 0 if array is invalid or empty
    if (!Array.isArray(values) || values.length === 0) return 0;
    
    // Sum all values using reduce()
    // reduce((a, b) => a + b, 0) adds all elements starting from 0
    const sum = values.reduce((a, b) => a + b, 0);
    
    // Return average (sum divided by count)
    return sum / values.length;
}

/**
 * Format byte count into human-readable string
 * 
 * Converts bytes to appropriate unit (B, KB, MB, GB)
 * 
 * @param {number} bytes - Byte count to format
 * @returns {string} Formatted string (e.g., "1.5 MB")
 */
function formatBytes(bytes) {
    // Handle zero bytes case
    if (bytes === 0) return '0 B';
    
    // Conversion factor (binary, not decimal)
    const k = 1024;
    
    // Unit labels in order of size
    const sizes = ['B', 'KB', 'MB', 'GB'];
    
    // Calculate which unit to use
    // log base k of bytes gives the unit index
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    // Convert to appropriate unit and format to 2 decimal places
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration into human-readable string
 * 
 * Converts seconds and nanoseconds to appropriate format
 * 
 * @param {number} sec - Seconds component
 * @param {number} nsec - Nanoseconds component
 * @returns {string} Formatted string (e.g., "5.23s" or "2m 30s")
 */
function formatDuration(sec, nsec) {
    // Convert to total seconds (nanoseconds are 1e-9 seconds)
    const total = sec + (nsec / 1e9);
    
    // Format based on duration length
    if (total < 1) {
        // Less than 1 second - show milliseconds
        return `${(total * 1000).toFixed(0)}ms`;
    } else if (total < 60) {
        // Less than 1 minute - show seconds
        return `${total.toFixed(2)}s`;
    } else {
        // 1 minute or more - show minutes and seconds
        const minutes = Math.floor(total / 60);    // Integer minutes
        const seconds = total % 60;                 // Remaining seconds
        return `${minutes}m ${seconds.toFixed(0)}s`;
    }
}
