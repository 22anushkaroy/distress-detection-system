// API Base URL - Uses relative path so it works in both local and production environments
const API_URL = "/api";

/**
 * Utility to render results dynamically.
 * @param {HTMLElement} containerElement - The DOM element to render inside
 * @param {Object} data - Expected to have { predicted_activity?, confidence?, is_danger, reason }
 * @param {String} overrideActivity - Hardcoded activity display option
 */
function displayResult(containerElement, data, overrideActivity = null) {
    containerElement.classList.remove('hidden');
    containerElement.innerHTML = '';

    // If there's an error from the API
    if (data.error) {
        containerElement.innerHTML = `<div class="error-msg">❌ Error: ${data.error}</div>`;
        return;
    }

    const activityText = overrideActivity || data.predicted_activity || "N/A";
    const confidenceText = data.confidence !== undefined && data.confidence !== null 
        ? `${(data.confidence * 100).toFixed(1)}%` 
        : "N/A";
    const reasonText = data.reason || "Processed normally";

    // Build the grid
    let html = `<div class="result-grid">`;
    
    // Predicted Activity
    html += `
        <div class="result-item">
            <div class="result-label">Predicted Activity</div>
            <div class="result-value">${activityText}</div>
        </div>
    `;

    // Confidence
    html += `
        <div class="result-item">
            <div class="result-label">Confidence</div>
            <div class="result-value">${confidenceText}</div>
        </div>
    `;

    // Reason
    html += `
        <div class="result-item" style="grid-column: span 2;">
            <div class="result-label">Reason</div>
            <div class="result-value" style="font-weight: 400; font-size: 0.95rem;">${reasonText}</div>
        </div>
    `;
    html += `</div>`;

    // Result Banner (SAFE or ALERT)
    if (data.is_danger) {
        html += `<div class="status-alert">🚨 ALERT — Distress Detected</div>`;
    } else {
        html += `<div class="status-safe">✅ SAFE — No Distress</div>`;
    }

    containerElement.innerHTML = html;
}

/**
 * 1. Sensor Detection
 */
async function detectSensor() {
    const activitySelect = document.getElementById('sensor-activity').value;
    const resultBox = document.getElementById('sensor-result');
    
    resultBox.classList.remove('hidden');
    resultBox.innerHTML = '<p style="text-align: center; color: #6b7280;">Processing sensor data...</p>';

    try {
        const response = await fetch(`${API_URL}/sensor`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ activity: activitySelect })
        });
        const data = await response.json();
        displayResult(resultBox, data);
    } catch (err) {
        console.error(err);
        displayResult(resultBox, { error: "Failed to connect to backend API." });
    }
}

/**
 * 2. Text Detection
 */
async function checkText() {
    const inputMsg = document.getElementById('text-input').value.trim();
    const resultBox = document.getElementById('text-result');

    if (!inputMsg) {
        alert("Please enter a message to check.");
        return;
    }

    resultBox.classList.remove('hidden');
    resultBox.innerHTML = '<p style="text-align: center; color: #6b7280;">Analyzing text...</p>';

    try {
        const response = await fetch(`${API_URL}/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: inputMsg })
        });
        const data = await response.json();
        // Custom formatting for text to map to standard UI display
        const displayData = {
            predicted_activity: data.keyword ? `Keyword: "${data.keyword}"` : "No keyword",
            confidence: null, // text doesn't always have confidence
            reason: data.reason || (data.is_danger ? "Distress keyword found" : "No distress words detected"),
            is_danger: data.is_danger
        };
        displayResult(resultBox, displayData);
    } catch (err) {
        console.error(err);
        displayResult(resultBox, { error: "Failed to connect to backend API." });
    }
}

/**
 * 3. Voice Detection
 */
async function analyzeVoice() {
    const fileInput = document.getElementById('voice-file');
    const resultBox = document.getElementById('voice-result');

    if (fileInput.files.length === 0) {
        alert("Please upload a .wav file first.");
        return;
    }

    const file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith('.wav')) {
        alert("Only .wav files are supported.");
        return;
    }

    resultBox.classList.remove('hidden');
    resultBox.innerHTML = '<p style="text-align: center; color: #6b7280;">Analyzing audio file (this may take a moment)...</p>';

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_URL}/voice`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        // Custom formatting for voice
        const displayData = {
            predicted_activity: data.recognized_text ? `Spoke: "${data.recognized_text}"` : "No speech recognized",
            confidence: null,
            reason: data.keyword ? `Trigger word matched: "${data.keyword}"` : "Clean audio",
            is_danger: data.is_danger
        };
        displayResult(resultBox, displayData);
    } catch (err) {
        console.error(err);
        displayResult(resultBox, { error: "Failed to connect to backend API." });
    }
}
