import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import base64


nama = "Dileando Gamaliel"
nim = "672021245"
univ = "Universitas Kristen Satya Wacana Salatiga"

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; padding: 10px; background-color: #f9f9fb; border-bottom: 1px solid #ddd;">
    <div style="flex-shrink: 0;">
        <img src="data:image/png;base64,{base64.b64encode(open('logo_univ.png', 'rb').read()).decode()}" width="100" />
    </div>
    <div style="flex-grow: 1; text-align: left; margin-left: 10px;">
        <h2 style="margin: 0; color: #435da3;">{univ}</h2>      
    </div>
</div>
""", unsafe_allow_html=True)
st.set_page_config(page_title="Deteksi Emosi")

st.title("Aplikasi Deteksi Emosi")
st.write("Aplikasi ini menggunakan DeepFace untuk mendeteksi emosi wajah.")

# --- SIDEBAR UNTUK MEMILIH MODE ---
mode = st.sidebar.selectbox("Pilih Mode", ["Webcam Real-Time", "Upload Gambar"])

# Fungsi untuk memproses gambar (digunakan baik untuk webcam maupun upload)
def deteksi_emosi(frame):
    try:
        # Analisis DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        screen_height, screen_width, _ = frame.shape #digunakan untuk menentukan ukuran frame

        for data_wajah in result:
            region = data_wajah['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Gambar kotak
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #membuat kotak berwarna di wajah
        
            # Ambil data emosi yang ada pada deepface
            emotion = data_wajah['dominant_emotion']
            score = data_wajah['emotion'][emotion]
            text_display = f"{emotion} ({int(score)}%)"

            # Tulis teks
            cv2.putText(frame, text_display, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    except Exception as e:
        return frame


# --- MODE 1: WEBCAM REAL-TIME ---
if mode == "Webcam Real-Time":
    st.subheader("Mode Kamera")
    st.write("Klik tombol 'Take Photo' di bawah untuk mendeteksi emosi.")

    # Menggunakan fitur native Streamlit untuk kamera di browser
    img_file_buffer = st.camera_input("Ambil Foto Wajah")

    if img_file_buffer is not None:
        # 1. Baca file gambar dari buffer
        bytes_data = img_file_buffer.getvalue()
        
        # 2. Konversi ke format OpenCV
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        st.write("Menganalisis...")
        
        # 3. Panggil fungsi deteksi yang sudah Anda buat
        frame_terproses = deteksi_emosi(cv2_img)

        # 4. Tampilkan hasil (Konversi BGR ke RGB)
        frame_rgb = cv2.cvtColor(frame_terproses, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Hasil Deteksi Real-Time", use_container_width=True)

# --- MODE 2: UPLOAD GAMBAR ---
elif mode == "Upload Gambar":
    st.subheader("Mode Upload Gambar")
    
    # Matikan kamera jika pindah ke mode upload (untuk keamanan resource)
    st.session_state['run'] = False 

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Konversi file upload ke format yang bisa dibaca OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.write("Sedang menganalisis...")
        
        # Panggil fungsi deteksi
        image_terproses = deteksi_emosi(image)

        # Konversi BGR ke RGB agar warna tampil benar di Streamlit
        image_rgb = cv2.cvtColor(image_terproses, cv2.COLOR_BGR2RGB)

        # Tampilkan hasil
        st.image(image_rgb, caption="Hasil Deteksi Emosi", use_container_width=True)

st.write("Tekan Icon Segitiga di pojok kiri untuk memilih mode upload gambar atau mode real time menggunakan webcam")

st.markdown("---")
st.markdown("""
### 📌 Dileando Gamaliel - 672021245


""", unsafe_allow_html=True)
