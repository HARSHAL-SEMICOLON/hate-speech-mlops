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
# 2. Enable CORS
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
    # Load ONNX model
    ort_session = ort.InferenceSession("model.onnx")

    # Load vectorizer
    with open("pytorch_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("✅ Model and Vectorizer loaded successfully.")

except Exception as e:
    print(f"❌ CRITICAL ERROR during startup: {e}")
    raise e


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

    try:
        # ----------------------------
        # Step A: Vectorize Text
        # ----------------------------

        text_vectorized = (
            vectorizer
            .transform([request.text])
            .toarray()
            .astype(np.float32)
        )

        # ----------------------------
        # Step B: Run ONNX Model
        # ----------------------------

        ort_inputs = {
            ort_session.get_inputs()[0].name: text_vectorized
        }

        ort_outs = ort_session.run(None, ort_inputs)

        raw_output = ort_outs[0]

        # Debug logs (very helpful on Render)
        print("RAW OUTPUT:", raw_output)
        print("OUTPUT SHAPE:", raw_output.shape)

        # ----------------------------
        # Step C: Extract Probability
        # ----------------------------

        if len(raw_output.shape) == 1:
            # Example: [0.87]
            probability = float(raw_output[0])

        elif raw_output.shape[1] == 1:
            # Example: [[0.87]]
            probability = float(raw_output[0][0])

        else:
            # Example: [[0.12, 0.88]]
            # Assume index 1 = Hate Speech
            probability = float(raw_output[0][1])

        # ----------------------------
        # Step D: Prediction Logic
        # ----------------------------

        prediction = (
            "Hate Speech"
            if probability > 0.5
            else "Safe"
        )

        # Confidence calculation
        conf_score = (
            probability
            if prediction == "Hate Speech"
            else (1 - probability)
        )

        confidence = f"{round(conf_score * 100, 2)}%"

        # ----------------------------
        # Step E: Response
        # ----------------------------

        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_score": probability
        }

    except Exception as e:

        print(f"⚠️ Inference Error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ================================
# 6. Health Check Endpoint
# ================================

@app.get("/")
async def root():

    return {
        "status": "Online",
        "engine": "ONNX Runtime",
        "version": "1.0.0"
    }