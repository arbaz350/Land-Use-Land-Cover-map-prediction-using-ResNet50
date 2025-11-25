import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------
st.title("🌍 Land Use / Land Cover (LULC) Classification")
st.write("Upload a JPG/PNG image to classify the land cover type.")

# ---------------------------------------------------------
# MODEL DOWNLOAD — HUGGINGFACE LINK
# ---------------------------------------------------------
MODEL_URL = "https://huggingface.co/arbajshaikh880/lulc-model/resolve/main/best_eurosat_128.keras"
MODEL_PATH = "best_eurosat_128.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... Please wait ⏳")

        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            st.success("Model downloaded successfully!")

        except Exception as e:
            st.error(f"Model download failed: {e}")
            st.stop()

download_model()

# ---------------------------------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_lulc_model():
    return load_model(MODEL_PATH)

model = load_lulc_model()

# ---------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------
class_labels = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# ---------------------------------------------------------
# PREPROCESS — JPG/PNG ONLY
# ---------------------------------------------------------
def preprocess_jpg(img):
    img = img.resize((128, 128))     # YOUR MODEL INPUT SIZE
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1,128,128,3)
    return img_array

# ---------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a JPG/PNG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_jpg(img)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")
    st.json({class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))})
