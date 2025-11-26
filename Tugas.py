#install opencv (pip install opencv-contrib-python)
import cv2
from deepface import DeepFace #install deepface (pip install deepface)

capture = cv2.VideoCapture(1) #memulai mengambil objek wajah dengan webcam

if not capture.isOpened():
    capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError("cannot open webcam")

while True:
    ret,frame = capture.read()
    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)

    data_wajah = result[0]
    screen_height, screen_width, _ = frame.shape
    region = data_wajah['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']

    if w < screen_width:
            
        # Gambar kotak di wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Tampilkan teks emosi 
        emotion = data_wajah['dominant_emotion']
        score = data_wajah['emotion'][emotion]
        text_display = f"{emotion} ({int(score)}%)"

        cv2.putText(frame, text_display, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

    else:
        # Jika wajah tidak ketemu (kotak full layar)
        cv2.putText(frame, "Wajah tidak jelas", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Original video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'): #membuat key untuk keluar dari webcam
        break

capture.release()
cv2.destroyAllWindows()