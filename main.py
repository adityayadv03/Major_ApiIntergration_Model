from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",   # your static HTML origin
    "http://localhost:5173",   # optional, if you also use Vite later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # or ["*"] while testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = tf.keras.models.load_model("model.h5")  # rename your h5 file to model.h5

# Image size used during training
IMG_SIZE = (224, 224)

# Class labels (same order as your training)
LABELS = [
    "Aloevera",
    "Curry",
    "Gotu Kola",
    "Hibiscus",
    "Lemon Grass",
    "Mint Leaf",
    "Moringa",
    "Neem",
    "Tulsi",
    "Turmeric"
]

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and convert image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor = preprocess_image(image)

    # Predict
    preds = model.predict(input_tensor)[0]
    confidence = float(np.max(preds))
    predicted_label = LABELS[np.argmax(preds)]

    # Low confidence warning (similar to your notebook logic)
    if confidence < 0.7:
        return {
            "status": "Uncertain",
            "message": "Possibly unidentified plant",
            "confidence": confidence
        }

    # Normal output
    return {
        "predicted_class": predicted_label,
        "confidence": confidence
    }
