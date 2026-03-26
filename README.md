# distress-detection-system
A simulation-based distress detection system using machine learning to analyze sensor data, text, and voice inputs for identifying abnormal human activity.
# 🛡️ SafeGuard: Distress Detection System

SafeGuard is a comprehensive, AI-powered distress detection platform designed to ensure student and personal safety in digital and physical environments. Built with a modern, glassmorphism-inspired purple UI, the system uses machine learning and pattern recognition to identify potential emergencies across three distinct vectors: **Sensor Data**, **Text Messages**, and **Voice Audio**.

![SafeGuard Theme](./frontend/hero-image.png)

## ✨ Features

*   **📱 Sensor Detection (Physical Activity)**
    *   Simulates and analyzes movement patterns (e.g., Normal Walking, Fall, Panic Running, Struggling).
    *   Uses a trained Machine Learning model (Random Forest via `scikit-learn`) to detect physical anomalies and trigger alerts.
*   **💬 Text Detection (Cyberbullying & Emergency)**
    *   Analyzes incoming text messages in real-time.
    *   Detects distress keywords, urgent requests for help, and potential threats to automatically flag dangerous situations.
*   **🎤 Voice Detection (Audio Analysis)**
    *   Accepts `.wav` voice uploads and processes speech-to-text using the `SpeechRecognition` library.
    *   Scans transcribed audio for emergency trigger words, providing a safety net for voice communications.
*   **🎨 Modern UI/UX**
    *   Beautiful, responsive frontend built with vanilla HTML/CSS/JS.
    *   Features a calming yet alert-ready purple/violet color palette, smooth animations, and a sticky glassmorphism navbar.

## 🛠️ Technology Stack

*   **Frontend:** Vanilla HTML5, CSS3 (CSS Variables, Flexbox/Grid, Animations), standard JavaScript (Fetch API).
*   **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python) for ultra-fast API routing and serving static files.
*   **Machine Learning & Data Processing:** `scikit-learn`, `numpy`, `pandas`.
*   **Audio Processing:** `SpeechRecognition`, `PyAudio`.

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/distress-project.git
    cd distress-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application (Single Service):**
    The project is configured so the FastAPI backend directly serves the frontend UI.
    ```bash
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
    ```

4.  **Access the app:**
    Open your browser and navigate to: `http://127.0.0.1:8000/`

## 📁 Project Structure

```text
distress-project/
├── api.py                 # FastAPI application entry point 
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
├── src/                   # Core detection logic
│   ├── model.py           # Random Forest ML model for sensor data
│   ├── anomaly.py         # Anomaly detection thresholds
│   ├── text_trigger.py    # Text parsing and keyword detection
│   └── voice_trigger.py   # Audio processing and speech-to-text
├── data/                  # Training data for ML models
└── frontend/              # User Interface
    ├── index.html         # Main dashboard
    ├── styles.css         # Purple theme and layout styles
    ├── script.js          # API connection logic
    └── hero-image.png     # UI illustration
