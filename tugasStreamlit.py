import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import base64


st.set_page_config(page_title="Deteksi Emosi")


nama = "Dileando Gamaliel"
nim = "672021245"
univ = "Universitas Kristen Satya Wacana Salatiga"


# try-except digunakan untuk mengecek file gambar
try:
    with open('logo_univ.png', 'rb') as f:
        logo_data = base64.b64encode(f.read()).decode()
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; padding: 10px; background-color: #f9f9fb; border-bottom: 1px solid #ddd;">
        <div style="flex-shrink: 0;">
            <img src="data:image/png;base64,{logo_data}" width="100" />
        </div>
        <div style="flex-grow: 1; text-align: left; margin-left: 10px;">
            <h2 style="margin: 0; color: #435da3;">{univ}</h2>      
        </div>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    # Fallback jika gambar logo tidak ada
    st.header(univ)

st.title("Aplikasi Deteksi Emosi")
st.write("Aplikasi ini menggunakan DeepFace untuk mendeteksi emosi wajah.")

# --- FUNGSI DETEKSI ---
def deteksi_emosi(frame):
    try:
        # DeepFace analyze
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        for data_wajah in result:
            region = data_wajah['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Gambar kotak hijau
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            # Ambil data emosi
            emotion = data_wajah['dominant_emotion']
            score = data_wajah['emotion'][emotion]
            text_display = f"{emotion} ({int(score)}%)"

            # Tulis teks emosi
            cv2.putText(frame, text_display, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    except Exception as e:
        # Jika tidak ada wajah terdeteksi atau error lain, kembalikan frame asli
        return frame

# --- SIDEBAR & PILIHAN MODE ---
mode = st.sidebar.selectbox("Pilih Mode", ["Ambil Foto (Webcam)", "Upload Gambar"])

# --- MODE 1: AMBIL FOTO (Pengganti Webcam Real-time) ---
if mode == "Ambil Foto (Webcam)":
    st.subheader("Mode Ambil Foto")
    st.info("Gunakan tombol di bawah untuk mengambil foto wajah Anda.")

    
    img_file_buffer = st.camera_input("Ambil Foto") #digunakan untuk mengambil gambar versi cloud 

    if img_file_buffer is not None:
        # 1. Baca data gambar
        bytes_data = img_file_buffer.getvalue()
        
        # 2. Decode ke format OpenCV
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        st.write("Menganalisis emosi...")
        
        # 3. Proses deteksi
        frame_terproses = deteksi_emosi(cv2_img)

        # 4. Tampilkan (Convert BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame_terproses, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Hasil Deteksi", use_container_width=True)

# --- MODE 2: UPLOAD GAMBAR ---
elif mode == "Upload Gambar":
    st.subheader("Mode Upload Gambar")
    
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. Baca data gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.write("Menganalisis emosi...")
        
        # 2. Proses deteksi
        image_terproses = deteksi_emosi(image)

        # 3. Tampilkan (Convert BGR -> RGB)
        image_rgb = cv2.cvtColor(image_terproses, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Hasil Deteksi Emosi", use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(f"### 📌 {nama} - {nim}")
