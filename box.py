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

# Utility functions
def enhance_contrast(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0

        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    image_np = np.array(image)
    image_np = enhance_contrast(image_np)  # Contrast-enhanced preprocessing

    input_h, input_w = model.input_shape[1], model.input_shape[2]

    # Preprocess based on input shape
    if model.input_shape[-1] == 1:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (input_w, input_h))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=(0, -1))
        processed_img = resized
    else:
        resized = cv2.resize(image_np, (input_w, input_h))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)
        processed_img = resized

    # Show original vs preprocessed
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(processed_img, caption="Preprocessed Image", use_column_width=True)

    # Predict
    prediction = model.predict(input_img)

    if prediction.shape[-1] == 1:
        confidence = float(prediction[0][0])
        class_idx = int(np.round(confidence))
        class_names = ["No Crack", "Crack"]
        probabilities = [1 - confidence, confidence]
    else:
        probabilities = prediction[0]
        class_idx = int(np.argmax(probabilities))
        class_names = [f"Class {i}" for i in range(len(probabilities))]

    label = class_names[class_idx]
    confidence_val = probabilities[class_idx]

    st.subheader(f"✅ Prediction: {label}")
    st.write(f"🔍 Model Confidence: {confidence_val:.2f}")
    st.progress(confidence_val)

    if len(probabilities) > 1:
        st.markdown("### 🔢 Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob:.2f}")

    # Grad-CAM Visualization
    st.markdown("### 🔍 Grad-CAM Visualization")
    last_conv_layer_name = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(input_img, model, last_conv_layer_name, class_idx)
    gradcam_output = overlay_heatmap(heatmap, processed_img)
    st.image(gradcam_output, caption="Grad-CAM", use_column_width=True)

    # Edge Detection
    st.markdown("### 🧱 Edge Detection")
    edges = cv2.Canny(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), 50, 150)
    st.image(edges, caption="Canny Edge Detection", use_column_width=True)

    # Save to session history
    st.session_state.history.append({
        "filename": uploaded_file.name,
        "prediction": label,
        "confidence": f"{confidence_val:.2f}"
    })

# Show prediction history
if st.session_state.history:
    st.markdown("### 📊 Prediction History")
    st.dataframe(st.session_state.history)
