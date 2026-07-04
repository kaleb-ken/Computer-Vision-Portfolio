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

# List of landmark connections
HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Wrist to thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Wrist to index
    (5, 9), (9, 10), (10, 11), (11, 12), # Index joint to Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Middle joint to Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Wrist to pinky
    (13, 17) # Pinky joint to Ring joint
]

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
    
    # Checking if left or right
    detected_L, detected_R = "", ""
    for i in range(len(result.hand_landmarks)):
        if result.handedness[i][0].category_name == "Left":
            detected_L = "detected"
        if result.handedness[i][0].category_name == "Right":
            detected_R = "detected"

    # Draws landmarks on detected hands
    hands_num = f"Hands detected: {len(result.hand_landmarks)}"
    if result.hand_landmarks:
            h, w, _ = frame.shape  # Get pixel dimensions of the camera feed
            for hand_landmarks in result.hand_landmarks:
                coord_landmarks = []
                # Gets landmark coords
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    coord_landmarks.append((cx,cy))
            
                # Draws lines between each landmark based
                for line in HAND_LINES:
                    list_start, list_end = line
                    point_start = coord_landmarks[list_start]
                    point_end = coord_landmarks[list_end]
    
                    cv2.line(frame, point_end, point_start, (203, 242, 172), 2)
                    cv2.circle(frame, coord_landmarks[list_end], 5, (191, 64, 191), cv2.FILLED)
    

    # Outputing text to feed
    cv2.putText(frame, hands_num, (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    cv2.putText(frame, f"Right: {detected_R}", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    cv2.putText(frame, f"Left: {detected_L}", (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)

   
    # Displaying feed
    cv2.imshow('capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()