import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('SurfaceCrackDetection.h5', compile=False)
    return model

model = load_model()

# App title and description
st.title("🧱 Concrete Crack Detection")
st.markdown("Upload an image of a concrete surface, and the AI model will determine whether there is a **crack** or **no crack**.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocessing
    image_np = np.array(image)
    input_h, input_w = model.input_shape[1], model.input_shape[2]

    # If grayscale input expected, convert and reshape
    if model.input_shape[-1] == 1:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (input_w, input_h))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, H, W, 1)
        processed_img = resized
    else:
        resized = cv2.resize(image_np, (input_w, input_h))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)  # Shape: (1, H, W, 3)
        processed_img = resized

    # Display preprocessed image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(processed_img, caption="Preprocessed", use_column_width=True)

    # Predict
    prediction = model.predict(input_img)
    
    # Handle binary or multiclass output
    if prediction.shape[-1] == 1:
        confidence = float(prediction[0][0])
        class_idx = int(np.round(confidence))
        class_names = ["No Crack", "Crack"]
        probabilities = [1 - confidence, confidence]
    else:
        probabilities = prediction[0]
        class_idx = int(np.argmax(probabilities))
        class_names = [f"Class {i}" for i in range(len(probabilities))]  # Customize if known

    label = class_names[class_idx]
    confidence_val = probabilities[class_idx]

    # Display prediction results
    st.subheader(f"✅ Prediction: {label}")
    st.write(f"🔍 Model Confidence: {confidence_val:.2f}")
    st.progress(confidence_val)

    # Show all class probabilities (if multiclass)
    if len(probabilities) > 1:
        st.markdown("### 🔢 Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob:.2f}")

   

    # Save to session history
    st.session_state.history.append({
        "filename": uploaded_file.name,
        "prediction": label,
        "confidence": f"{confidence_val:.2f}"
    })

# Show prediction history table
if st.session_state.history:
    st.markdown("### 📊 Prediction History")
    st.table(st.session_state.history)
