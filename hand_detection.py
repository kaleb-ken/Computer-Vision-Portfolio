"""
hand_detection.py
======================
Basic hand tracking program with landmark display

"""

# Adding Dependencies
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


# Setting up capture
feed = cv2.VideoCapture(0)
feed.set(cv2.CAP_PROP_FRAME_HEIGHT, value=500)
feed.set(cv2.CAP_PROP_FRAME_WIDTH, value=500)

# Setting up hand detection
model_path = "./models/hand_landmarker.task"
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

start_time = time.time()

# Running video feed
while True:
    ret, frame = feed.read()
    if not ret:
        break

    # Converting colour config for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting hand in video
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(mp_image,timestamp_ms)
    
    # Outputing text to feed
    text = f"Hands detected: {len(result.hand_landmarks)}"
    cv2.putText(frame, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=15)
   
    # Displaying feed
    cv2.imshow('capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()