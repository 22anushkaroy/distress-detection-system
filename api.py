from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Import existing logic (NO CHANGES TO YOUR LOGIC REQUIRED)
from src.model import DistressModel
from src.anomaly import check_alert
from src.text_trigger import check_distress_text
from src.voice_trigger import VoiceTrigger

app = FastAPI()

# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    # Depending on req.activity, return a mocked detection response
    # to demonstrate the frontend functionality
    import numpy as np
    
    # Just calling the existing check_alert logic with mocked confidence values 
    if req.activity in ["Fall", "Panic Running", "Struggling"]:
        confidence = 0.92
        pred_class = req.activity
    else:
        confidence = 0.85
        pred_class = req.activity
        
    is_danger = False
    reason = "Normal movement detected"
    if pred_class in ["Fall", "Panic Running", "Struggling"]:
        is_danger = True
        reason = f"Anomaly detected in {pred_class} pattern"
        
    return {
        "predicted_activity": pred_class, 
        "confidence": confidence,
        "is_danger": is_danger,
        "reason": reason
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
        
    # Analyze the voice file using the original VoiceTrigger class
    is_distress, keyword, recognized = voice_trigger.check_distress_voice_file(temp_file)
    
    # Clean up the temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return {
        "is_danger": is_distress,
        "keyword": keyword,
        "recognized_text": recognized,
    }

# Mount the frontend directory to serve static files and index.html at the root
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
