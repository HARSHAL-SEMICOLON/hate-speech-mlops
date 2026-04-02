from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import pickle
import re

app = FastAPI(title="MLOps Hate Speech Detection (ONNX)")

# Allow your local HTML to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the ONNX Inference Session
print("Loading High-Performance ONNX Model...")
try:
    # This replaces the heavy torch.load()
    ort_session = ort.InferenceSession("model.onnx")
    vectorizer = pickle.load(open("pytorch_vectorizer.pkl", "rb"))
    print("✅ ONNX Model & Vectorizer Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class UserInput(BaseModel):
    text: str

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.post("/predict")
async def analyze_text(data: UserInput):
    cleaned = clean_text(data.text)
    
    # --- Step 1: Heuristic Safety Net ---
    positive_words = ["good", "great", "happy", "love", "wonderful", "beautiful", "nice", "awesome"]
    bad_words = ["hate", "kill", "stupid", "idiot", "ugly", "trash"]
    words_in_input = cleaned.split()
    
    if any(word in words_in_input for word in positive_words) and not any(word in words_in_input for word in bad_words):
        return {"prediction": "Safe", "confidence": 98.5}

    # --- Step 2: ONNX Inference ---
    # Convert text to the exact numerical format the model expects
    vectorized_text = vectorizer.transform([cleaned]).toarray().astype(np.float32)
    
    # Run the model through the ONNX Runtime engine
    ort_inputs = {ort_session.get_inputs()[0].name: vectorized_text}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # The output is a probability between 0 and 1
    probability = float(ort_outs[0][0])
    confidence_score = float(probability * 100)

    # --- Step 3: Threshold Logic ---
    if probability > 0.65:
        return {
            "prediction": "Hate Speech", 
            "confidence": round(confidence_score, 1)
        }
    else:
        # Invert score for "Safe" display (100 - probability)
        return {
            "prediction": "Safe", 
            "confidence": round(100 - confidence_score, 1)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)