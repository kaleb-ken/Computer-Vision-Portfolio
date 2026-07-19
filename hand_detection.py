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
import hand_functions.hand_data as hd
import hand_functions.kalman_filter as kf
#import hand_functions.hand_instruments as hi

# Set up optimization
kalman = kf.kalman()

# Setting up capture
model_input = None
feed = cv2.VideoCapture(0)
feed.set(cv2.CAP_PROP_FRAME_HEIGHT, value=500)
feed.set(cv2.CAP_PROP_FRAME_WIDTH, value=500)

# Setting up hand detection
model_path = "./models/gesture_recognizer.task"
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options, 
    num_hands=2, 
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8, 
    running_mode=vision.RunningMode.VIDEO)
detector = vision.GestureRecognizer.create_from_options(options)
start_time = time.time()

# Running video feed
while True:
    ret, frame = feed.read()
    if not ret:
        break
    # Initilize the canvas as a black image
    if model_input is None: 
        model_input = np.zeros_like(frame)
    
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converting colour config for mediapipe

    # Detecting hand in video
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.recognize_for_video(mp_image,timestamp_ms)
    
    # Kalman filter for optimised detection (Still working on)
    predicted = kalman.predict()
    pred_x, pred_y = int(predicted[0]), int(predicted[1])

    # Visualisations for detections
    hands_num = f"Hands detected: {len(result.hand_landmarks)}"
    hv.draw_hand_struct(result, frame) # Draws landmarks on detected hands
    hv.draw_hand_struct(result, model_input) # Draws landmarks on detected hands
    gesture = hv.detect_gesture(result)
    handedness = hv.handedness(result)

    # ----------Code for arduino----------
    #hi.buzz_detection(handedness)

    # Outputing text to feed
    cv2.putText(frame, hands_num, (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    cv2.putText(frame, f"Gesture: {gesture}", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    
    # Displaying feed
    cv2.imshow('capture', frame) # Standard Video
    cv2.imshow('Model Input', model_input) # Standard Video
    model_input = None

    key = cv2.waitKey(5) & 0xFF # Detects keyboard input

    if key == ord('q'): # Quits application
        break
    if key == ord('t'): # Saves hand data as image and csv
        #model_input = hd.screenshot_hand(frame, result)
        hd.save_landmark_data(result)


    

feed.release()
cv2.destroyAllWindows()