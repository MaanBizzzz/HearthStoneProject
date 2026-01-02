import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS as tts
from playsound import playsound
import os

model = tf.keras.models.load_model("asl_cnn_model_rel.h5")
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

last_label = None
current_label = None

def hand_callback(result, output_image, timestamp_ms):
    global last_label, current_label

    if not result.hand_landmarks:
        current_label = None
        return

    landmarks = result.hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    ref = pts[0]
    pts = pts - ref
    pts = pts[1:]
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts /= (scale + 1e-6)

    data = np.expand_dims(pts, axis=0)
    preds = model.predict(data, verbose=0)
    label = class_names[np.argmax(preds)]
    current_label = label

    if label != last_label:
        letter = tts(text=label, lang='en')
        letter.save("letter.mp3")
        playsound("letter.mp3", block=False)
        os.remove("letter.mp3")
        last_label = label

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=hand_callback
)

landmarker = HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)
timestamp_ms = 0

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)
    landmarker.detect_async(mp_image, timestamp_ms)
    timestamp_ms += 1

    if current_label:
        cv2.putText(frame, current_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Hand Sign Prediction", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
