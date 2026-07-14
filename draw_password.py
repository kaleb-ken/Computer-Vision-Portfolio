"""
hand_detection.py
======================
Free hand drawing application

"""

# Adding Dependencies
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import hand_functions.hand_visuals as hv
#import hand_functions.hand_instruments as hi

# Setting up capture
feed = cv2.VideoCapture(0)
feed.set(cv2.CAP_PROP_FRAME_HEIGHT, value=500)
feed.set(cv2.CAP_PROP_FRAME_WIDTH, value=500)

# Variables for drawing functionality
canvas = None
prev_point = None

# Setting up hand detection
model_path = "./models/gesture_recognizer.task"
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)
detector = vision.GestureRecognizer.create_from_options(options)
start_time = time.time()

# Running video feed
while True:
    ret, frame = feed.read()
    if not ret:
        break
    if canvas is None: # Initilize the canvas as a black image
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converting colour config for mediapipe

    # Detecting hand in video
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.recognize_for_video(mp_image,timestamp_ms)
    
    # Visualisations for detections
    hands_num = f"Hands detected: {len(result.hand_landmarks)}"
    hv.draw_hand_struct(result, frame) # Draws landmarks on detected hands
    gesture = hv.detect_gesture(result)

    # ----------Code for arduino----------
    #hi.buzz_detection(handedness)

    # ----------Code for drawing----------
    drawing, prev_point = hv.point_free_draw(result, frame, canvas, prev_point)
    if drawing is not None:
        frame = cv2.add(frame, drawing)
    else:
        canvas = None # Deletes drawing

    # Outputing text to feed
    cv2.putText(frame, hands_num, (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    cv2.putText(frame, f"Gesture: {gesture}", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    cv2.rectangle(frame, (100,100), (200,200), (255, 255, 255), 2, lineType=None)
    
    # Displaying feed
    cv2.imshow('capture', frame)
    
    key = cv2.waitKey(5) & 0xFF # Detects keyboard input

    if key == ord('t'):
        cv2.imwrite("filename.jpg", canvas)

    if key == ord('q'): # Quits application
        break

    

feed.release()
cv2.destroyAllWindows()