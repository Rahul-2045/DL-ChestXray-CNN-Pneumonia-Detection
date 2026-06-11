import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "TensorFlow is not installed.\n\n"
        "Please run this command in terminal:\n\n"
        "python -m pip install tensorflow"
    )
    st.stop()


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan CNN",
    page_icon="🫁",
    layout="centered"
)


# ---------------------------------------------------------
# Title
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
# Helper: Find model automatically
# ---------------------------------------------------------
def find_model_file():
    """
    This function searches model file in the same folder where app.py is present.
    It checks common model names first, then searches any .keras or .h5 file.
    """
    current_folder = Path(__file__).parent

    possible_names = [
        "pneumonia_cnn_model.keras",
        "pneumonia_cnn_model.h5",
        "best_pneumonia_model.keras",
        "best_pneumonia_model.h5",
        "model.keras",
        "model.h5"
    ]

    for name in possible_names:
        model_path = current_folder / name
        if model_path.exists():
            return str(model_path)

    for ext in ["*.keras", "*.h5"]:
        files = list(current_folder.glob(ext))
        if files:
            return str(files[0])

    return ""


# ---------------------------------------------------------
# Helper: Show folder files for debugging
# ---------------------------------------------------------
def show_current_folder_files():
    current_folder = Path(__file__).parent
    files = [file.name for file in current_folder.iterdir() if file.is_file()]
    return current_folder, files


# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
st.sidebar.header("Model Settings")

auto_model_path = find_model_file()

model_path = st.sidebar.text_input(
    "Model file path",
    value=auto_model_path,
    help="Paste full model path here if model is not detected automatically."
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
# Debug Section
# ---------------------------------------------------------
with st.sidebar.expander("Check app folder"):
    folder, files = show_current_folder_files()
    st.write("App folder:")
    st.code(str(folder))

    st.write("Files available in app folder:")
    if files:
        st.write(files)
    else:
        st.write("No files found.")

    st.write("Detected model path:")
    st.code(auto_model_path if auto_model_path else "No model detected")


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_model(path):
    return tf.keras.models.load_model(path)


if not model_path:
    st.error(
        "Model file not found automatically.\n\n"
        "Please save your trained model in the same folder as this app file.\n\n"
        "Recommended model names:\n"
        "- pneumonia_cnn_model.keras\n"
        "- pneumonia_cnn_model.h5\n\n"
        "Or paste the full model path in the sidebar."
    )

    st.info(
        "To save model from notebook, run:\n\n"
        'model.save(r"E:\\Rahul Verma\\document\\D drive\\PROJECTS\\Vscode\\Deeplearning\\pneumonia_cnn_model.keras")'
    )
    st.stop()


if not os.path.exists(model_path):
    st.error(
        f"Model file not found:\n\n{model_path}\n\n"
        "Please check the file name/path or paste the correct full path in the sidebar."
    )
    st.stop()


try:
    model = load_cnn_model(model_path)
    st.success(f"Model loaded successfully: {model_path}")
except Exception as e:
    st.error("Model file found, but could not be loaded.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------
# Image Preprocessing
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
        raw_score = float(prediction[0][0])

        if raw_score > threshold:
            predicted_label = "PNEUMONIA"
            confidence = raw_score
        else:
            predicted_label = "NORMAL"
            confidence = 1 - raw_score

        st.subheader("Prediction Result")

        if predicted_label == "PNEUMONIA":
            st.error(f"Prediction: {predicted_label}")
        else:
            st.success(f"Prediction: {predicted_label}")

        st.write(f"Confidence Score: **{confidence:.2%}**")
        st.write(f"Raw Model Score: `{raw_score:.6f}`")

        st.progress(min(max(confidence, 0.0), 1.0))

        st.info(
            "Prediction logic: If raw model score is greater than threshold, "
            "prediction is PNEUMONIA; otherwise prediction is NORMAL."
        )

else:
    st.info("Please upload a chest X-ray image to start prediction.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Project: Chest X-Ray Pneumonia Detection using CNN | TensorFlow + Streamlit")
