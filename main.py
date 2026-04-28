from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os
import urllib.request

app = FastAPI()

# ✅ Allow Flutter / external requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔥 MODEL CONFIG
# =========================

MODEL_PATH = "hybridcropnet.onnx"

# 🔥 REPLACE THIS WITH YOUR GOOGLE DRIVE DIRECT LINK
MODEL_URL = "https://drive.google.com/file/d/1Ii-GxBIHjpMr9HWkZpYsYyWgzNLa7oLt/view?usp=sharing"

# ✅ Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# ✅ Load model
session = ort.InferenceSession(MODEL_PATH)

# =========================
# 🌿 CLASS NAMES
# =========================

class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Healthy",
    "Pepper Bacterial Spot"
]

# =========================
# 🧠 PREPROCESSING
# =========================

def preprocess(image):
    image = image.resize((224, 224))  # adjust if needed
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# 🟢 TEST ROUTE
# =========================

@app.get("/")
def home():
    return {"message": "AgroSense AI Backend Running 🚀"}

# =========================
# 🔥 PREDICT API
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 📷 Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 🧠 Preprocess
        input_tensor = preprocess(image)

        # 🤖 Run model
        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)

        probs = outputs[0][0]
        predicted_index = int(np.argmax(probs))
        confidence = float(probs[predicted_index])

        disease = class_names[predicted_index]

        # 🌾 Dummy yield prediction (replace later with ML model)
        yield_prediction = round(5 + confidence * 5, 2)

        return {
            "disease": disease,
            "confidence": confidence,
            "yield": yield_prediction
        }

    except Exception as e:
        return {"error": str(e)}
