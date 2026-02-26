// Nomos Dashboard - Application
// Control Plane: http://127.0.0.1:8081

const CONFIG = {
    controlPlane: 'http://127.0.0.1:8081',
    refreshInterval: 2000,
    maxLogEntries: 100
};

// State
let state = {
    connected: false,
    metrics: {
        latencyP99: 0,
        throughputRps: 0,
        successRate: 0,
        healedRequests: 0,
        totalRequests: 0
    },
    logs: [],
    healingHistory: []
};

// DOM Elements
const elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    setupNavigation();
    setupEventListeners();
    initializeCharts();
    startMetricsRefresh();
    loadInitialData();
});

function cacheElements() {
    elements.navItems = document.querySelectorAll('.nav-item');
    elements.sections = document.querySelectorAll('.section');
    elements.statusDot = document.getElementById('status-dot');
    elements.connectionText = document.getElementById('connection-text');
    elements.refreshBtn = document.getElementById('refresh-btn');
    elements.lastUpdated = document.getElementById('last-updated');
    
    // Metrics
    elements.latencyValue = document.getElementById('latency-value');
    elements.rpsValue = document.getElementById('rps-value');
    elements.successValue = document.getElementById('success-value');
    elements.healedValue = document.getElementById('healed-value');
    
    // NLP
    elements.nlpSourceField = document.getElementById('nlp-source-field');
    elements.nlpTargetField = document.getElementById('nlp-target-field');
    elements.nlpCompareBtn = document.getElementById('nlp-compare-btn');
    
    // Logs
    elements.logContainer = document.getElementById('log-entries');
    elements.clearLogsBtn = document.getElementById('clear-logs');
    elements.logFilterSelect = document.getElementById('log-filter');
    
    // Settings
    elements.healingEnabled = document.getElementById('healing-enabled');
    elements.confidenceThreshold = document.getElementById('confidence-threshold');
    elements.thresholdValue = document.getElementById('threshold-value');
}

function setupNavigation() {
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => {
            const target = item.dataset.target;
            
            // Update nav
            elements.navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Update sections
            elements.sections.forEach(s => s.classList.remove('active'));
            document.getElementById(`${target}-section`).classList.add('active');
        });
    });
}

function setupEventListeners() {
    // Refresh button
    if (elements.refreshBtn) {
        elements.refreshBtn.addEventListener('click', refreshMetrics);
    }
    
    // NLP Compare
    if (elements.nlpCompareBtn) {
        elements.nlpCompareBtn.addEventListener('click', runNlpComparison);
    }
    
    // Clear logs
    if (elements.clearLogsBtn) {
        elements.clearLogsBtn.addEventListener('click', clearLogs);
    }
    
    // Log filter
    if (elements.logFilterSelect) {
        elements.logFilterSelect.addEventListener('change', filterLogs);
    }
    
    // Confidence threshold slider
    if (elements.confidenceThreshold) {
        elements.confidenceThreshold.addEventListener('input', (e) => {
            elements.thresholdValue.textContent = (e.target.value / 100).toFixed(2);
        });
    }
    
    // Healing enabled toggle
    if (elements.healingEnabled) {
        elements.healingEnabled.addEventListener('change', (e) => {
            addLog('info', `Healing ${e.target.checked ? 'enabled' : 'disabled'}`);
        });
    }
}

// Metrics
async function fetchMetrics() {
    try {
        const response = await fetch(`${CONFIG.controlPlane}/health`);
        if (!response.ok) throw new Error('Control plane unreachable');
        
        const data = await response.json();
        updateConnectionStatus(true);
        return data;
    } catch (error) {
        updateConnectionStatus(false);
        return null;
    }
}

function updateConnectionStatus(connected) {
    state.connected = connected;
    
    if (elements.statusDot) {
        elements.statusDot.className = `status-dot ${connected ? 'connected' : 'error'}`;
    }
    if (elements.connectionText) {
        elements.connectionText.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

async function refreshMetrics() {
    const metrics = await fetchMetrics();
    
    if (metrics) {
        updateMetricsUI(metrics);
    } else {
        // Demo data when not connected
        updateMetricsUI(generateDemoMetrics());
    }
    
    if (elements.lastUpdated) {
        elements.lastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
    }
}

function updateMetricsUI(data) {
    if (elements.latencyValue) {
        elements.latencyValue.textContent = `${data.latency_p99 || state.metrics.latencyP99}ms`;
    }
    if (elements.rpsValue) {
        elements.rpsValue.textContent = `${data.throughput_rps || state.metrics.throughputRps}`;
    }
    if (elements.successValue) {
        elements.successValue.textContent = `${data.success_rate || state.metrics.successRate}%`;
    }
    if (elements.healedValue) {
        elements.healedValue.textContent = formatNumber(data.healed_requests || state.metrics.healedRequests);
    }
    
    state.metrics = { ...state.metrics, ...data };
}

function generateDemoMetrics() {
    return {
        latency_p99: (Math.random() * 0.5 + 0.15).toFixed(2),
        throughput_rps: Math.floor(Math.random() * 2000 + 4000),
        success_rate: (99 + Math.random()).toFixed(2),
        healed_requests: Math.floor(Math.random() * 1000 + 12000)
    };
}

function startMetricsRefresh() {
    refreshMetrics();
    setInterval(refreshMetrics, CONFIG.refreshInterval);
}

// Charts
let latencyChart = null;
let throughputChart = null;
let latencyData = [];
let throughputData = [];

function initializeCharts() {
    // Generate initial chart bars
    for (let i = 0; i < 30; i++) {
        latencyData.push(Math.random() * 0.3 + 0.1);
        throughputData.push(Math.random() * 2000 + 4000);
    }
    
    renderSimpleChart('latency-chart', latencyData, 1);
    renderSimpleChart('throughput-chart', throughputData, 6000);
}

function renderSimpleChart(canvasId, data, maxValue) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const barWidth = (width / data.length) - 2;
    
    // Clear
    ctx.clearRect(0, 0, width, height);
    
    // Draw bars
    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, '#1d9bf0');
    gradient.addColorStop(1, '#00ba7c');
    
    data.forEach((value, index) => {
        const barHeight = (value / maxValue) * height * 0.9;
        const x = index * (barWidth + 2);
        const y = height - barHeight;
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, [4, 4, 0, 0]);
        ctx.fill();
    });
}

function updateCharts() {
    // Add new data points
    latencyData.push(Math.random() * 0.3 + 0.1);
    throughputData.push(Math.random() * 2000 + 4000);
    
    // Keep last 30 points
    if (latencyData.length > 30) latencyData.shift();
    if (throughputData.length > 30) throughputData.shift();
    
    renderSimpleChart('latency-chart', latencyData, 1);
    renderSimpleChart('throughput-chart', throughputData, 6000);
}

setInterval(updateCharts, 1000);

// NLP Comparison
function runNlpComparison() {
    const source = elements.nlpSourceField?.value || 'user_name';
    const target = elements.nlpTargetField?.value || 'username';
    
    if (!source || !target) {
        addLog('warn', 'Please enter both source and target field names');
        return;
    }
    
    // Calculate similarities using different algorithms
    const results = {
        levenshtein: calculateLevenshtein(source, target),
        jaroWinkler: calculateJaroWinkler(source, target),
        ngram: calculateNGram(source, target),
        soundex: calculateSoundexSimilarity(source, target),
        metaphone: calculateMetaphoneSimilarity(source, target)
    };
    
    // Find best match
    const ensemble = calculateEnsemble(results);
    
    // Update UI
    updateNlpResults(results, ensemble);
    
    addLog('info', `NLP comparison: "${source}" vs "${target}" - confidence: ${(ensemble * 100).toFixed(1)}%`);
}

function updateNlpResults(results, ensemble) {
    // Update result bars
    Object.entries(results).forEach(([algo, score]) => {
        const fill = document.querySelector(`#result-${algo} .result-fill`);
        const value = document.querySelector(`#result-${algo} .result-value`);
        
        if (fill) {
            fill.style.width = `${score * 100}%`;
        }
        if (value) {
            value.textContent = (score * 100).toFixed(1) + '%';
        }
    });
    
    // Update ensemble
    const ensembleFill = document.querySelector('#result-ensemble .result-fill');
    const ensembleValue = document.querySelector('#result-ensemble .result-value');
    
    if (ensembleFill) {
        ensembleFill.style.width = `${ensemble * 100}%`;
    }
    if (ensembleValue) {
        ensembleValue.textContent = (ensemble * 100).toFixed(1) + '%';
    }
}

// NLP Algorithm Implementations

function calculateLevenshtein(s1, s2) {
    s1 = s1.toLowerCase();
    s2 = s2.toLowerCase();
    
    const m = s1.length;
    const n = s2.length;
    
    if (m === 0) return n === 0 ? 1 : 0;
    if (n === 0) return 0;
    
    const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
    
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
            dp[i][j] = Math.min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            );
        }
    }
    
    const maxLen = Math.max(m, n);
    return 1 - dp[m][n] / maxLen;
}

function calculateJaroWinkler(s1, s2) {
    s1 = s1.toLowerCase();
    s2 = s2.toLowerCase();
    
    if (s1 === s2) return 1;
    
    const len1 = s1.length;
    const len2 = s2.length;
    
    if (len1 === 0 || len2 === 0) return 0;
    
    const matchWindow = Math.floor(Math.max(len1, len2) / 2) - 1;
    const s1Matches = new Array(len1).fill(false);
    const s2Matches = new Array(len2).fill(false);
    
    let matches = 0;
    let transpositions = 0;
    
    for (let i = 0; i < len1; i++) {
        const start = Math.max(0, i - matchWindow);
        const end = Math.min(i + matchWindow + 1, len2);
        
        for (let j = start; j < end; j++) {
            if (s2Matches[j] || s1[i] !== s2[j]) continue;
            s1Matches[i] = true;
            s2Matches[j] = true;
            matches++;
            break;
        }
    }
    
    if (matches === 0) return 0;
    
    let k = 0;
    for (let i = 0; i < len1; i++) {
        if (!s1Matches[i]) continue;
        while (!s2Matches[k]) k++;
        if (s1[i] !== s2[k]) transpositions++;
        k++;
    }
    
    const jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3;
    
    // Winkler modification
    let prefix = 0;
    for (let i = 0; i < Math.min(4, Math.min(len1, len2)); i++) {
        if (s1[i] === s2[i]) prefix++;
        else break;
    }
    
    return jaro + prefix * 0.1 * (1 - jaro);
}

function calculateNGram(s1, s2, n = 2) {
    s1 = s1.toLowerCase();
    s2 = s2.toLowerCase();
    
    if (s1.length < n || s2.length < n) {
        return s1 === s2 ? 1 : 0;
    }
    
    const getNgrams = (s) => {
        const ngrams = new Set();
        for (let i = 0; i <= s.length - n; i++) {
            ngrams.add(s.substring(i, i + n));
        }
        return ngrams;
    };
    
    const ngrams1 = getNgrams(s1);
    const ngrams2 = getNgrams(s2);
    
    let intersection = 0;
    ngrams1.forEach(ng => {
        if (ngrams2.has(ng)) intersection++;
    });
    
    const union = ngrams1.size + ngrams2.size - intersection;
    return union === 0 ? 0 : intersection / union;
}

function calculateSoundexSimilarity(s1, s2) {
    const soundex = (s) => {
        s = s.toUpperCase().replace(/[^A-Z]/g, '');
        if (s.length === 0) return '0000';
        
        const map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        };
        
        let result = s[0];
        let prev = map[s[0]] || '0';
        
        for (let i = 1; i < s.length && result.length < 4; i++) {
            const code = map[s[i]];
            if (code && code !== prev) {
                result += code;
            }
            prev = code || prev;
        }
        
        return (result + '0000').substring(0, 4);
    };
    
    const code1 = soundex(s1);
    const code2 = soundex(s2);
    
    let matches = 0;
    for (let i = 0; i < 4; i++) {
        if (code1[i] === code2[i]) matches++;
    }
    
    return matches / 4;
}

function calculateMetaphoneSimilarity(s1, s2) {
    // Simplified Double Metaphone
    const metaphone = (s) => {
        s = s.toUpperCase().replace(/[^A-Z]/g, '');
        if (s.length === 0) return '';
        
        let result = '';
        let i = 0;
        
        // Skip initial silent letters
        if (['KN', 'GN', 'PN', 'AE', 'WR'].some(p => s.startsWith(p))) {
            i = 1;
        }
        
        while (i < s.length && result.length < 4) {
            const c = s[i];
            const next = s[i + 1] || '';
            
            switch (c) {
                case 'A': case 'E': case 'I': case 'O': case 'U':
                    if (i === 0) result += c;
                    break;
                case 'B':
                    if (i !== s.length - 1 || s[i - 1] !== 'M') result += 'P';
                    break;
                case 'C':
                    if (next === 'H') { result += 'X'; i++; }
                    else if ('EIY'.includes(next)) result += 'S';
                    else result += 'K';
                    break;
                case 'D':
                    if (next === 'G' && 'EIY'.includes(s[i + 2] || '')) { result += 'J'; i += 2; }
                    else result += 'T';
                    break;
                case 'G':
                    if ('EIY'.includes(next)) result += 'J';
                    else result += 'K';
                    break;
                case 'H':
                    if (!'AEIOU'.includes(s[i - 1] || '') || 'AEIOU'.includes(next)) result += 'H';
                    break;
                case 'K':
                    if (s[i - 1] !== 'C') result += 'K';
                    break;
                case 'P':
                    if (next === 'H') { result += 'F'; i++; }
                    else result += 'P';
                    break;
                case 'Q': result += 'K'; break;
                case 'S':
                    if (next === 'H') { result += 'X'; i++; }
                    else result += 'S';
                    break;
                case 'T':
                    if (next === 'H') { result += '0'; i++; }
                    else result += 'T';
                    break;
                case 'W': case 'Y':
                    if ('AEIOU'.includes(next)) result += c;
                    break;
                case 'X': result += 'KS'; break;
                case 'Z': result += 'S'; break;
                default:
                    if ('FJLMNR'.includes(c)) result += c;
            }
            i++;
        }
        
        return result;
    };
    
    const code1 = metaphone(s1);
    const code2 = metaphone(s2);
    
    if (code1 === code2) return 1;
    if (code1.length === 0 || code2.length === 0) return 0;
    
    // Calculate similarity between metaphone codes
    return calculateLevenshtein(code1, code2);
}

function calculateEnsemble(results) {
    const weights = {
        levenshtein: 0.25,
        jaroWinkler: 0.30,
        ngram: 0.20,
        soundex: 0.10,
        metaphone: 0.15
    };
    
    let weighted = 0;
    Object.entries(results).forEach(([algo, score]) => {
        weighted += score * (weights[algo] || 0.2);
    });
    
    return weighted;
}

// Logs
function addLog(level, message) {
    const timestamp = new Date().toLocaleTimeString();
    const entry = { timestamp, level, message };
    
    state.logs.unshift(entry);
    if (state.logs.length > CONFIG.maxLogEntries) {
        state.logs.pop();
    }
    
    renderLogs();
}

function renderLogs() {
    if (!elements.logContainer) return;
    
    const filter = elements.logFilterSelect?.value || 'all';
    const filteredLogs = filter === 'all' 
        ? state.logs 
        : state.logs.filter(l => l.level === filter);
    
    elements.logContainer.innerHTML = filteredLogs.map(log => `
        <div class="log-entry ${log.level}">
            <span class="log-time">${log.timestamp}</span>
            <span class="log-level">${log.level.toUpperCase()}</span>
            <span class="log-message">${log.message}</span>
        </div>
    `).join('');
}

function filterLogs() {
    renderLogs();
}

function clearLogs() {
    state.logs = [];
    renderLogs();
    addLog('info', 'Logs cleared');
}

// Initial Data
function loadInitialData() {
    // Add some initial logs
    addLog('info', 'Dashboard initialized');
    addLog('info', 'Connecting to control plane...');
    
    // Load demo healing history
    state.healingHistory = [
        { timestamp: '12:34:56', original: 'user_name', healed: 'username', operation: 'rename', confidence: 0.95 },
        { timestamp: '12:34:57', original: 'created_date', healed: 'createdAt', operation: 'rename', confidence: 0.88 },
        { timestamp: '12:34:58', original: '"123"', healed: '123', operation: 'coerce', confidence: 0.99 }
    ];
    
    // Initial NLP comparison
    setTimeout(() => {
        runNlpComparison();
    }, 500);
}

// Utilities
function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        calculateLevenshtein,
        calculateJaroWinkler,
        calculateNGram,
        calculateSoundexSimilarity,
        calculateMetaphoneSimilarity,
        calculateEnsemble
    };
}
