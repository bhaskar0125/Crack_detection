import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('SurfaceCrackDetection.h5', compile=False)
    return model

model = load_model()

st.title("Concrete Crack Detection")
st.write("Upload an image of a concrete surface to detect cracks.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess: grayscale, resize, normalize
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Dynamically resize to model's expected input size
    input_h, input_w = model.input_shape[1], model.input_shape[2]
    resized = cv2.resize(gray, (input_w, input_h))
    normalized = resized / 255.0
    input_img = np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, H, W, 1)
    # If model expects 3 channels, stack grayscale
    if model.input_shape[-1] == 3:
        input_img = np.repeat(input_img, 3, axis=-1)  # (1, H, W, 3)
    
    # Predict
    prediction = model.predict(input_img)
    class_idx = int(np.round(prediction[0][0])) if prediction.shape[-1] == 1 else np.argmax(prediction)
    label = "Crack" if class_idx == 1 else "No Crack"
    st.subheader(f"Prediction: {label}")
    st.write(f"Model confidence: {float(prediction[0][0]) if prediction.shape[-1]==1 else float(np.max(prediction)):.2f}")
    
