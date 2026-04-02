from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

# ================================
# 1. Initialize FastAPI
# ================================
app = FastAPI(title="NLP Moderation Core API")

# ================================
# 2. Enable CORS (Critical for Netlify)
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 3. Load Model + Vectorizer
# ================================
try:
    print("🔄 Initializing Neural Engine...")
    
    # Load ONNX session
    ort_session = ort.InferenceSession("model.onnx")

    # Load the vectorizer (TF-IDF/Count)
    with open("pytorch_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("✅ System Online: Model and Vectorizer Loaded.")
except Exception as e:
    print(f"❌ CRITICAL STARTUP ERROR: {e}")
    # We don't raise here to allow the process to stay alive for logs
    ort_session = None
    vectorizer = None

# ================================
# 4. Request Schema
# ================================
class TextRequest(BaseModel):
    text: str

# ================================
# 5. Prediction Endpoint
# ================================
@app.post("/predict")
async def analyze_text(request: TextRequest):
    if ort_session is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded on server.")

    try:
        # Step A: Vectorize Input
        text_vectorized = (
            vectorizer
            .transform([request.text])
            .toarray()
            .astype(np.float32)
        )

        # Step B: Run ONNX Inference
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: text_vectorized}
        ort_outs = ort_session.run(None, ort_inputs)

        # The raw array from the model
        raw_output = ort_outs[0]

        # ---------------------------------------------------------
        # Step C: SAFE EXTRACTION (The Fix for Python 3.14)
        # ---------------------------------------------------------
        # .flatten() ensures we have a 1D array [val1, val2...]
        # .item() converts a 1-element array into a true Python float
        flat_output = raw_output.flatten()
        
        if len(flat_output) > 1:
            # If model returns [Safe_Prob, Hate_Prob], we take the 2nd value
            probability = float(flat_output[1].item())
        else:
            # If model returns just [Hate_Prob]
            probability = float(flat_output[0].item())

        # ---------------------------------------------------------
        # Step D: Classification Logic
        # ---------------------------------------------------------
        if probability > 0.5:
            prediction = "Hate Speech"
            confidence_val = probability
        else:
            prediction = "Safe"
            confidence_val = 1 - probability

        confidence_str = f"{round(confidence_val * 100, 2)}%"

        return {
            "prediction": prediction,
            "confidence": confidence_str,
            "raw_score": float(probability),
            "status": "Success"
        }

    except Exception as e:
        print(f"⚠️ Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# 6. Health Check
# ================================
@app.get("/")
async def root():
    return {
        "status": "Online",
        "engine": "ONNX Runtime",
        "version": "1.0.1",
        "author": "Harshal"
    }