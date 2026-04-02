from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

# 1. Setup FastAPI
app = FastAPI(title="Hate Speech Detection API")

# 2. Enable CORS (Vital for Netlify)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Global variables for Model and Vectorizer
ort_session = None
vectorizer = None

# 4. Load Artifacts on Startup
try:
    print("🔄 Loading AI Model and Vectorizer...")
    ort_session = ort.InferenceSession("model.onnx")
    with open("pytorch_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ System Ready.")
except Exception as e:
    print(f"❌ Startup Error: {e}")

class TextRequest(BaseModel):
    text: str

# 5. Prediction Logic
@app.post("/predict")
async def analyze_text(request: TextRequest):
    if ort_session is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Step A: Transform input text
        text_vectorized = vectorizer.transform([request.text]).toarray().astype(np.float32)

        # Step B: Run Inference
        input_name = ort_session.get_inputs()[0].name
        ort_outs = ort_session.run(None, {input_name: text_vectorized})

        # --- THE ABSOLUTE FIX FOR LINE 61 ---
        # We flatten the output to a 1D list and use .item()
        # .item() is the ONLY way to satisfy Python 3.14's strict scalar rules
        raw_output = ort_outs[0].flatten()
        
        # If model has 2 outputs [Safe, Hate], take index 1. 
        # If it has 1 output [Hate], take index 0.
        prob_index = 1 if len(raw_output) > 1 else 0
        probability = raw_output[prob_index].item() 

        # Step D: Result Formatting
        prediction = "Hate Speech" if probability > 0.5 else "Safe"
        confidence_val = probability if prediction == "Hate Speech" else (1 - probability)
        
        return {
            "prediction": prediction,
            "confidence": f"{round(confidence_val * 100, 2)}%",
            "raw_score": float(probability)
        }

    except Exception as e:
        print(f"⚠️ Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "Online"}