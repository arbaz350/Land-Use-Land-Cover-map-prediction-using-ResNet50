import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

st.title("LULC Classification")
st.write("Upload an image to classify Land Use / Land Cover")

# -------------------------------
# DOWNLOAD MODEL IF NOT PRESENT
# -------------------------------
model_path = "lulc_50_epoch.keras"
drive_id = "1h_1yv_bgfhCh2SY64GZzWv2pYo31fHpO"  # your file ID
drive_url = f"https://drive.google.com/uc?id={drive_id}"

if not os.path.exists(model_path):
    st.write("Downloading model… Please wait (250MB).")
    gdown.download(drive_url, model_path, quiet=False)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model(model_path)

# Class labels
class_labels = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'Sealake'
]

# -------------------------------
# FILE UPLOAD SECTION
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((64, 64))  # match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
