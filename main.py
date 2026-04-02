from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hate Speech Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts
ort_session = None
vectorizer = None

@app.on_event("startup")
def load_artifacts():
    global ort_session, vectorizer
    try:
        ort_session = ort.InferenceSession("model.onnx")
        with open("pytorch_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ ARTIFACTS LOADED SUCCESSFULLY")
    except Exception as e:
        print(f"❌ STARTUP ERROR: {e}")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def analyze_text(request: TextRequest):
    try:
        # Preprocess
        vec = vectorizer.transform([request.text]).toarray().astype(np.float32)
        
        # Inference
        inputs = {ort_session.get_inputs()[0].name: vec}
        outs = ort_session.run(None, inputs)
        
        # 🚀 THE CRITICAL FIX FOR LINE 61
        # We use .item() which is the only way to convert 1-element arrays in Python 3.14
        raw_val = outs[0].flatten()
        print(f"DEBUG - Raw Output Shape: {raw_val.shape}")
        
        # Extract based on output size (handles [prob] or [safe_prob, hate_prob])
        probability = float(raw_val[-1].item()) 

        prediction = "Hate Speech" if probability > 0.5 else "Safe"
        conf = probability if prediction == "Hate Speech" else (1 - probability)

        return {
            "prediction": prediction,
            "confidence": f"{round(conf * 100, 2)}%",
            "status": "success"
        }
    except Exception as e:
        print(f"❌ INFERENCE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "version": "2.0.1_final_fix"}