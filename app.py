import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Rice Leaf Disease Detection")
st.write("Upload a photo of a rice leaf to identify the disease.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_leaf_model.keras")

model = load_model()
labels = {v: k for k, v in json.load(open("class_names.json")).items()}

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    x = preprocess_input(np.array(img.resize((224, 224)), dtype="float32")[None])
    probs = model.predict(x)[0]

    st.success(f"Prediction: {labels[probs.argmax()]} ({probs.max():.0%} confidence)")
    st.bar_chart({labels[i]: float(p) for i, p in enumerate(probs)})

    st.caption("Prototype trained on 119 lab-style images with plain backgrounds. Results on field photos may be less accurate.")