from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

# ✅ Allow Flutter / external requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development (later restrict)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 Load model once (VERY IMPORTANT for real-time)
session = ort.InferenceSession("quantized_model.onnx")

# ⚠️ Update based on your model classes
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Healthy",
    "Pepper Bacterial Spot"
]

# 🧠 Preprocessing
def preprocess(image):
    image = image.resize((224, 224))  # change if your model differs
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img

# 🟢 Test route
@app.get("/")
def home():
    return {"message": "AgroSense AI Backend Running 🚀"}

# 🔥 REAL-TIME PREDICTION API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_tensor = preprocess(image)

        # Run model
        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)

        probs = outputs[0][0]
        predicted_index = int(np.argmax(probs))
        confidence = float(probs[predicted_index])

        disease = class_names[predicted_index]

        # 🌾 Simple yield logic (replace later)
        yield_prediction = round(5 + confidence * 5, 2)

        return {
            "disease": disease,
            "confidence": confidence,
            "yield": yield_prediction
        }

    except Exception as e:
        return {"error": str(e)}