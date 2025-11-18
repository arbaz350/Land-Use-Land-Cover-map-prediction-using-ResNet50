import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import os

st.title("🌍 Land Use / Land Cover (LULC) Classification")
st.write("Upload an image to classify the land cover type using a deep learning model.")

# --------------------------------------------------
# MODEL DOWNLOAD FROM HUGGING FACE (RECOMMENDED)
# --------------------------------------------------

MODEL_URL = "https://huggingface.co/arbajshaikh880/lulc-model/blob/main/lulc_50_epoch.keras"
MODEL_PATH = "lulc_50_epoch.keras"

def download_model():
    """Download model from HuggingFace if not already available."""
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... (250MB) Please wait ⏳")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model download completed!")
        else:
            st.error("Failed to download model. Check the HuggingFace URL.")
            st.stop()

download_model()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_lulc_model():
    return load_model(MODEL_PATH)

model = load_lulc_model()

# Class labels (your classes)
class_labels = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'Sealake'
]

# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((64, 64))  # Match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
