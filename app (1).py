import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan CNN",
    page_icon="🫁",
    layout="centered"
)


# ---------------------------------------------------------
# App Title
# ---------------------------------------------------------
st.title("🫁 PneumoScan: Chest X-Ray Pneumonia Detection")
st.write(
    "Upload a chest X-ray image and the CNN model will predict whether it is "
    "**NORMAL** or **PNEUMONIA**."
)

st.warning(
    "This app is for learning and project demonstration only. "
    "It should not be used for real medical diagnosis."
)


# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
st.sidebar.header("Model Settings")

default_model_path = "pneumonia_cnn_model.h5"

model_path = st.sidebar.text_input(
    "Model file path",
    value=default_model_path,
    help="Keep pneumonia_cnn_model.h5 in the same folder as app.py or provide full path."
)

image_size = st.sidebar.selectbox(
    "Image size used during training",
    options=[150, 224],
    index=0
)

threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

IMAGE_SIZE = (image_size, image_size)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_model(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


model = load_cnn_model(model_path)

if model is None:
    st.error(
        f"Model file not found: {model_path}\n\n"
        "Please keep `pneumonia_cnn_model.h5` in the same folder as `app.py`, "
        "or provide the correct model path in the sidebar."
    )
    st.stop()

st.success("Model loaded successfully.")


# ---------------------------------------------------------
# Image Preprocessing Function
# ---------------------------------------------------------
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)

    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array


# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original_img, img_array = preprocess_image(uploaded_file)

    st.subheader("Uploaded X-Ray Image")
    st.image(original_img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        confidence_score = float(prediction[0][0])

        if confidence_score > threshold:
            predicted_label = "PNEUMONIA"
            display_confidence = confidence_score
        else:
            predicted_label = "NORMAL"
            display_confidence = 1 - confidence_score

        st.subheader("Prediction Result")

        if predicted_label == "PNEUMONIA":
            st.error(f"Prediction: {predicted_label}")
        else:
            st.success(f"Prediction: {predicted_label}")

        st.write(f"Confidence Score: **{display_confidence:.2%}**")
        st.write(f"Raw Model Score: `{confidence_score:.6f}`")

        st.progress(min(max(display_confidence, 0.0), 1.0))

        st.info(
            "Model logic: If raw score is greater than threshold, prediction is PNEUMONIA; "
            "otherwise prediction is NORMAL."
        )

else:
    st.info("Please upload a chest X-ray image to start prediction.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Project: Chest X-Ray Pneumonia Detection using CNN | Built with TensorFlow and Streamlit")
