# Distress Detection UI

This is a clean, minimal UI for your Distress Detection System. It separates the frontend from your Python logic and uses simple HTML, CSS, and Vanilla JavaScript.

## How to Run

### 1. The Frontend (UI)
Since it's basic HTML/JS, you can run it using any simple web server. 
If you have Python installed, open your terminal in the `frontend` folder and run:
```bash
python -m http.server 3000
```
Then, open your browser and go to: `http://localhost:3000`

### 2. The Backend (API calls)
The UI expects a backend API running at `http://127.0.0.1:8000`. 
Currently, your `main.py` uses Streamlit. To connect this new UI to your backend logic without changing the core logic, you can create a simple **FastAPI** wrapper (e.g., `api.py`).

Here is how you can expose your existing logic to the UI:

```python
# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing logic (NO CHANGES TO YOUR LOGIC REQUIRED)
from src.model import DistressModel
from src.anomaly import check_alert
from src.text_trigger import check_distress_text
from src.voice_trigger import VoiceTrigger

app = FastAPI()

# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model globally
model = DistressModel()
model.train('data/train', 'data/my_sensor_data')
voice_trigger = VoiceTrigger()

class SensorRequest(BaseModel):
    activity: str

class TextRequest(BaseModel):
    text: str

@app.post("/api/sensor")
async def sensor_endpoint(req: SensorRequest):
    # Depending on req.activity, generate mock data and pass to model
    # (Just like in your Streamlit main.py)
    # Return JSON:
    return {
        "predicted_activity": "Fall", 
        "confidence": 0.95,
        "is_danger": True,
        "reason": "Anomaly detected in fall pattern"
    }

@app.post("/api/text")
async def text_endpoint(req: TextRequest):
    is_distress, keyword = check_distress_text(req.text)
    return {
        "is_danger": is_distress,
        "keyword": keyword,
        "reason": "Trigger word found" if is_distress else "No distress words"
    }

@app.post("/api/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())
        
    is_distress, keyword, recognized = voice_trigger.check_distress_voice_file(temp_file)
    
    return {
        "is_danger": is_distress,
        "keyword": keyword,
        "recognized_text": recognized,
    }
```

**To run the API:**
1. Install FastAPI and Uvicorn: `pip install fastapi uvicorn`
2. Run the server: `uvicorn api:app --reload`

Now your clean HTML UI will successfully communicate with your robust Python backend!
