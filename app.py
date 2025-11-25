import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import os

# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------
st.title("🌍 Land Use / Land Cover (LULC) Classification")
st.write("Upload an image to classify the land cover type using a deep learning model.")

# ---------------------------------------------------------
# MODEL URL — REPLACE WITH YOUR REAL HUGGINGFACE URL
# ---------------------------------------------------------
MODEL_URL = "https://huggingface.co/arbajshaikh880/lulc-model/resolve/main/best_eurosat_128.keras"
MODEL_PATH = "best_eurosat_128.keras"


# ---------------------------------------------------------
# SAFE CHUNK-STREAM DOWNLOAD (FIXES CORRUPTION)
# ---------------------------------------------------------
def download_model():
    """Stream download the model from HuggingFace to avoid corruption."""
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... This may take 1–3 minutes (250MB). Please wait ⏳")

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


# ---------------------------------------------------------
# DOWNLOAD MODEL IF MISSING
# ---------------------------------------------------------
download_model()


# ---------------------------------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_lulc_model():
    try:
        return load_model(MODEL_PATH)
    except Exception:
        st.error("❌ Failed to load model. The file may be corrupted.")
        st.stop()


model = load_lulc_model()


# ---------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------
class_labels = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'Sealake'
]


# ---------------------------------------------------------
# IMAGE UPLOADER
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((64, 64,13))  # Match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
