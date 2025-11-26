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
    
    # Inisialisasi state kamera
    if 'run' not in st.session_state:
        st.session_state['run'] = False #kondisi awal masuk web harus dalam keadaan kamera mati

    def start_camera():
        st.session_state['run'] = True

    def stop_camera():
        st.session_state['run'] = False

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Nyalakan Kamera", on_click=start_camera)
    with col2:
        stop_btn = st.button("Matikan Kamera", on_click=stop_camera)

    frame_window = st.image([])
    
    if st.session_state['run']:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened(): #percabangan untuk menentukan kondisi nilai dari kamera
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                st.error("Kamera tidak ditemukan!")
        
        else:
            while st.session_state['run']:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal membaca frame.")
                    break
                
                # Panggil fungsi deteksi
                frame_terproses = deteksi_emosi(frame)

                # Konversi BGR ke RGB untuk Streamlit
                frame_rgb = cv2.cvtColor(frame_terproses, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb)
            
            cap.release()
    else:
        st.info("Tekan tombol untuk memulai kamera.")

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