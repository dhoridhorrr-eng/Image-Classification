import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import re
import gdown
import zipfile  # Library untuk mengekstrak file ZIP

st.set_page_config(page_title="Klasifikasi Retak Beton")

# --- KONFIGURASI GOOGLE DRIVE ---
GOOGLE_DRIVE_SHARE_LINK = "https://drive.google.com/file/d/16RT_dahvxqh0VeYFdWS-UTBlE2YK1g3C/view?usp=sharing"

MODEL_PATH = "model_crack_beton.h5"
CLASS_NAMES = ["Retak", "Tidak Retak"]
IMG_HEIGHT = 150
IMG_WIDTH = 150

def extract_gdrive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if match:
        return match.group(1)
    return None

@st.cache_resource
def load_model():
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
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Klasifikasi Retak Beton")
st.write("Upload gambar satuan atau file ZIP berisi kumpulan gambar beton.")

# Mengubah tipe file yang diizinkan agar mendukung .zip
uploaded_file = st.file_uploader(
    "Pilih file gambar atau file .zip",
    type=["jpg", "jpeg", "png", "zip"]
)

# Fungsi pembantu untuk memproses dan memprediksi satu gambar PIL
def predict_image(img, file_name):
    # 1. Pastikan ukuran sesuai target model
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img_resized)
    
    # 2. UJI COBA PREPROCESSING (Deteksi otomatis jika nilai stuck)
    # Kita buat duplikat array untuk mencoba prediksi tanpa pembagian 255
    img_array_raw = np.expand_dims(img_array, axis=0) # Skala 0-255
    img_array_norm = np.expand_dims(img_array / 255.0, axis=0) # Skala 0-1
    
    # Coba prediksi menggunakan skala 0-1 dulu
    prediction = model.predict(img_array_norm, verbose=0)
    
    # JIKA HASILNYA MASIH STUCK DI NILAI YANG SAMA (Contoh nilai mentahnya konstan)
    # Kita beralih menggunakan data mentah (0-255) tanpa normalisasi
    if np.allclose(prediction, prediction[0][0], atol=1e-4):
        prediction = model.predict(img_array_raw, verbose=0)

    # 3. LOGIKA PENENTUAN SKOR DAN PERSENTASE KELAS
    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        
        # Logika dinamis: Menghitung jarak dari threshold 0.5
        if score >= 0.5:
            # Jika hasil ini terbalik pada tes Anda, tukar string "Retak" dan "Tidak Retak"
            predicted_class = "Tidak Retak"
            confidence = score * 100
        else:
            predicted_class = "Retak"
            confidence = (1 - score) * 100
    else:
        # Untuk model output Softmax (2 komponen output)
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        idx = np.argmax(probabilities)
        
        classes_map = {0: "Tidak Retak", 1: "Retak"}
        predicted_class = classes_map.get(idx, "Tidak Diketahui")
        confidence = float(probabilities[idx]) * 100
        
    return predicted_class, confidence

# --- PROSES UPLOAD ---
if uploaded_file is not None:
    # JIKA FILE YANG DIUPLOAD ADALAH ZIP
    if uploaded_file.name.endswith('.zip'):
        st.info(f"Mengekstrak file ZIP: {uploaded_file.name}...")
        
        # Buka file zip di dalam memori
        with zipfile.ZipFile(uploaded_file) as z:
            # Ambil semua daftar file di dalam zip
            file_list = z.namelist()
            
            # Saring hanya file yang berupa gambar
            valid_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in file_list if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX')]
            
            if not image_files:
                st.error("Tidak ditemukan file gambar (.jpg, .png) di dalam file ZIP tersebut.")
            else:
                st.success(f"Ditemukan {len(image_files)} gambar. Memulai proses klasifikasi...")
                
                # Loop untuk memproses setiap gambar di dalam ZIP
                for file_path in image_files:
                    # Ambil nama filenya saja tanpa struktur folder dalam zip
                    base_name = os.path.basename(file_path)
                    if base_name == "": continue 
                    
                    with z.open(file_path) as f:
                        try:
                            img = Image.open(f).convert("RGB")
                            
                            # Tampilkan expander untuk menghemat ruang di layar Streamlit
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
                                    st.write(f"Kepercayaan: {confidence:.2f}%")
                        except Exception as e:
                            st.warning(f"Gagal membaca file {base_name}: {e}")

    # JIKA FILE YANG DIUPLOAD ADALAH GAMBAR SATUAN (.jpg, .png)
    else:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
        
        predicted_class, confidence = predict_image(img, uploaded_file.name)
        st.success(f"Hasil: {predicted_class}")
        st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
