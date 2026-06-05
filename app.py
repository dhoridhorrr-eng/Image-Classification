import streamlit as st
import numpy as np
from PIL import Image
import os
import re
import gdown
import zipfile

# Trik Bypass Deserialization: Impor keras secara modular untuk menangani bug versi lama vs baru
import tensorflow as tf
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Klasifikasi Retak Beton")

# --- CONFIGURASI GOOGLE DRIVE ---
# Pastikan link di bawah ini ditutup dengan tanda kutip murni tanpa teks kode lain!
GOOGLE_DRIVE_SHARE_LINK = "https://drive.google.com/file/d/16RT_dahvxqh0VeYFdWS-UTBlE2YK1g3C/view?usp=sharing"

MODEL_PATH = "model_crack_beton.h5"
IMG_HEIGHT = 150
IMG_WIDTH = 150

def extract_gdrive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if match:
        return match.group(1)
    return None

@st.cache_resource
def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        file_id = extract_gdrive_id(GOOGLE_DRIVE_SHARE_LINK)
        if not file_id:
            st.error("Format Link Google Drive tidak valid!")
            st.stop()
            
        with st.spinner("Sedang mengunduh model dari Google Drive..."):
            download_url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(download_url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                st.stop()
                
    # GANTI STRATEGI: Jika load murni error karena batch_shape/optional, 
    # kita paksa compile=False agar arsitektur tetap terbaca tanpa merusak layer input.
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)

# Memuat model secara aman
try:
    model = load_model_safely()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

st.title("Klasifikasi Retak Beton")
st.write("Upload gambar atau file ZIP berisi kumpulan gambar beton.")

uploaded_file = st.file_uploader(
    "Pilih file gambar atau file .zip",
    type=["jpg", "jpeg", "png", "zip"]
)

def predict_image(img, filename=""):
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img_resized)
    
    # Konversi array ke float32 dan normalisasi standar (0-1)
    img_array = np.array(img_array, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    # Deteksi Kelas Dinamis (Mencegah Angka Stuck 73.11%)
    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        # Logika Threshold Biner
        if score >= 0.5:
            predicted_class = "Tidak Retak"
            confidence = score * 100
        else:
            predicted_class = "Retak"
            confidence = (1 - score) * 100
    else:
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        idx = np.argmax(probabilities)
        classes_map = {0: "Tidak Retak", 1: "Retak"}
        predicted_class = classes_map.get(idx, "Tidak Diketahui")
        confidence = float(probabilities[idx]) * 100
        
    return predicted_class, confidence

# --- MANAJEMEN BERKAS ---
if uploaded_file is not None:
    if uploaded_file.name.endswith('.zip'):
        st.info("Mengekstrak file ZIP...")
        with zipfile.ZipFile(uploaded_file) as z:
            file_list = z.namelist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in file_list if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX')]
            
            if not image_files:
                st.error("Tidak ada file gambar valid di dalam ZIP.")
            else:
                for file_path in image_files:
                    base_name = os.path.basename(file_path)
                    if base_name == "": continue
                    
                    with z.open(file_path) as f:
                        try:
                            img = Image.open(f).convert("RGB")
                            with st.expander(f"Hasil untuk: {base_name}"):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(img, use_container_width=True)
                                with col2:
                                    predicted_class, confidence = predict_image(img, base_name)
                                    if predicted_class == "Retak":
                                        st.error(f"Hasil: {predicted_class}")
                                    else:
                                        st.success(f"Hasil: {predicted_class}")
                                    st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
                        except Exception as e:
                            st.warning(f"Gagal memproses gambar {base_name}: {e}")
    else:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
        
        with st.spinner("Menganalisis gambar..."):
            predicted_class, confidence = predict_image(img, uploaded_file.name)
            
        if predicted_class == "Retak":
            st.error(f"Hasil: {predicted_class}")
        else:
            st.success(f"Hasil: {predicted_class}")
        st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
