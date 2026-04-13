# -------------------------------
# Imports (CLEANED)
# -------------------------------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import predict
import gdown
import os

# -------------------------------
# Page Config (MUST BE FIRST STREAMLIT COMMAND)
# -------------------------------
st.set_page_config(page_title="AI Vision", layout="wide", page_icon="🤖")

# -------------------------------
# Download model from Google Drive
# -------------------------------
MODEL_PATH = "model/cat_dog_model.h5"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1nbNFrN91xeUJ3L4TR_1mvyeCWfRz1CZU"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load Model (ONLY ONCE ✅)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------
# Custom CSS (Premium UI)
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(120deg, #0f172a, #1e293b);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Sidebar text WHITE */
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Title */
.main-title {
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card */
.card {
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    backdrop-filter: blur(20px);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-6px) scale(1.01);
}

/* Upload */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,0.3);
    border-radius: 15px;
    padding: 10px;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    border-radius: 10px;
    color: white;
}

/* Footer */
.footer {
    text-align:center;
    opacity:0.6;
    margin-top:40px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ IMAGE CLASSIFIER")
st.sidebar.write("Upload an image to classify")
st.sidebar.info("Model: CNN | Cats vs Dogs")

# -------------------------------
# Main Header
# -------------------------------
st.markdown('<div class="main-title">AI Vision Classifier</div>', unsafe_allow_html=True)
st.write("### Classify images with AI in real-time")

# Layout
col1, col2 = st.columns([1, 1])

# -------------------------------
# LEFT: Upload
# -------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RIGHT: Prediction
# -------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing image..."):
            prediction, confidence = predict(model, image)

        # Score
        st.metric("Prediction Score", f"{prediction:.4f}")

        # Progress
        st.progress(int(confidence * 100))

        # Probabilities
        st.write(f"🐱 Cat Probability: {(1 - prediction) * 100:.2f}%")
        st.write(f"🐶 Dog Probability: {prediction * 100:.2f}%")

        # Result
        if prediction > 0.5:
            st.success(f"🐶 Dog ({confidence*100:.2f}% confidence)")
        else:
            st.success(f"🐱 Cat ({confidence*100:.2f}% confidence)")

    else:
        st.info("Upload an image to see prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown('<div class="footer">Built with ❤️ | AI Internship Project</div>', unsafe_allow_html=True)
