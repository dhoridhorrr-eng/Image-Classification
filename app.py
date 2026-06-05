import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Klasifikasi Retak Beton")

MODEL_PATH = "model_crack_beton.h5"
CLASS_NAMES = ["Retak", "Tidak Retak"]

IMG_HEIGHT = 150
IMG_WIDTH = 150

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Klasifikasi Retak Beton")
st.write("Upload gambar beton untuk mendeteksi retak atau tidak retak.")

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        predicted_class = CLASS_NAMES[0] if score >= 0.5 else CLASS_NAMES[1]
        confidence = max(score, 1 - score) * 100
    else:
        score = tf.nn.softmax(prediction[0])
        idx = np.argmax(score)
        predicted_class = CLASS_NAMES[idx]
        confidence = float(np.max(score)) * 100

    st.success(f"Hasil: {predicted_class}")
    st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
