import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown  # Tambahkan library ini untuk mendownload dari Google Drive

st.set_page_config(page_title="Klasifikasi Retak Beton")

# --- CONFIGURASI GOOGLE DRIVE ---
# 1. Pastikan file .h5 di Google Drive Anda sudah di-share ke "Anyone with the link" (Siapa saja yang memiliki link)
# 2. Ambil ID file dari link share tersebut dan masukkan ke bawah ini.
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/file/d/16RT_dahvxqh0VeYFdWS-UTBlE2YK1g3C/view?usp=sharing"
MODEL_PATH = "model_crack_beton.h5"

CLASS_NAMES = ["Retak", "Tidak Retak"]
IMG_HEIGHT = 150
IMG_WIDTH = 150

@st.cache_resource
def load_model():
    # Jika model belum ada di server Streamlit, download otomatis dari Google Drive
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Sedang mengunduh model dari Google Drive (ini hanya dilakukan sekali)..."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Google Drive: {e}")
                st.stop()
                
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di path: {MODEL_PATH}")
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
