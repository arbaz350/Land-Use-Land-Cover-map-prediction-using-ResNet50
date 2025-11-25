import streamlit as st
import numpy as np
from PIL import Image
import rasterio
import requests
import os
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------
st.title("🌍 Land Use / Land Cover (LULC) Classification")
st.write("Upload a EuroSAT **TIF (13-band)** satellite image to classify land cover type.")

# ---------------------------------------------------------
# MODEL URL (HuggingFace)
# ---------------------------------------------------------
MODEL_URL = "https://huggingface.co/arbajshaikh880/lulc-model/resolve/main/best_eurosat_128.keras"
MODEL_PATH = "best_eurosat_128.keras"

# ---------------------------------------------------------
# DOWNLOAD MODEL IF NOT EXIST
# ---------------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... please wait ⏳")
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
# CLASS LABELS (EuroSAT 10 classes)
# ---------------------------------------------------------
class_labels = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# ---------------------------------------------------------
# READ & PREPROCESS TIF (13 Bands)
# ---------------------------------------------------------
def preprocess_tif(file, target_size=(128, 128)):
    with rasterio.open(file) as src:
        img = src.read()  # shape: (13, H, W)

        # Center crop (EuroSAT images vary)
        h, w = img.shape[1], img.shape[2]
        crop = min(h, w)
        start_h = (h - crop) // 2
        start_w = (w - crop) // 2
        img = img[:, start_h:start_h + crop, start_w:start_w + crop]

        # Resize each band → (128,128)
        bands = []
        for b in img:
            b_img = Image.fromarray(b.astype(np.uint8))
            b_img = b_img.resize(target_size)
            bands.append(np.array(b_img))

        img = np.stack(bands, axis=-1)  # (128,128,13)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, 0)  # (1,128,128,13)

# ---------------------------------------------------------
# UI — FILE UPLOADER
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a **TIF** satellite image...", type=["tif", "tiff"])

if uploaded_file is not None:
    st.success("TIF file uploaded!")

    # Preview using RGB Bands (1,2,3)
    try:
        with rasterio.open(uploaded_file) as src:
            rgb = src.read([1, 2, 3])
            rgb = np.moveaxis(rgb, 0, -1)
            st.image(rgb, caption="RGB Preview", use_column_width=True)
    except:
        st.warning("No RGB preview available")

    # Preprocess
    img = preprocess_tif(uploaded_file)

    #
