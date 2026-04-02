from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
try:
    ort_session = ort.InferenceSession("model.onnx")
    with open("pytorch_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Startup Error: {e}")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    try:
        # Vectorize
        vec = vectorizer.transform([request.text]).toarray().astype(np.float32)
        
        # Inference
        out = ort_session.run(None, {ort_session.get_inputs()[0].name: vec})
        
        # THE FIX: Aggressive flattening and item extraction
        # This converts ANY array shape into a simple Python float
        prob = float(np.array(out[0]).flatten()[0])

        prediction = "Hate Speech" if prob > 0.5 else "Safe"
        conf = prob if prob > 0.5 else (1 - prob)

        return {
            "prediction": prediction,
            "confidence": f"{round(conf * 100, 2)}%",
            "raw_score": prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online"}