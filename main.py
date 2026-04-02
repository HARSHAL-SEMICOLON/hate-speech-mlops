from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI(title="Hate Speech Detection API")

# 2. Enable CORS (Allows your Netlify site to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the ONNX Model and Vectorizer
try:
    # Load ONNX session
    ort_session = ort.InferenceSession("model.onnx")
    
    # Load the vectorizer (ensure this file is in your GitHub repo)
    with open("pytorch_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model artifacts: {e}")

# 4. Define Request Model
class TextRequest(BaseModel):
    text: str

# 5. Prediction Logic
@app.post("/predict")
async def predict(request: TextRequest):
    try:
        # Preprocess text using the loaded vectorizer
        text_vectorized = vectorizer.transform([request.text]).toarray().astype(np.float32)

        # Run ONNX Inference
        ort_inputs = {ort_session.get_inputs()[0].name: text_vectorized}
        ort_outs = ort_session.run(None, ort_inputs)

        # FIX: Safely extract the scalar probability value
        # This handles the 'TypeError: only 0-dimensional arrays' bug
        probability = float(ort_outs[0].flatten()[0])

        # Classification threshold
        prediction = "Hate Speech" if probability > 0.5 else "Safe"
        confidence = round(probability * 100, 2) if prediction == "Hate Speech" else round((1 - probability) * 100, 2)

        return {
            "prediction": prediction,
            "confidence": f"{confidence}%",
            "raw_score": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Root Health Check
@app.get("/")
async def root():
    return {"status": "Online", "model": "ONNX Moderation Core"}