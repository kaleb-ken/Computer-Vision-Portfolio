"""
hand_visuals.py
=======================
Functions for hand detection script.
Just to help decompose code
"""

# Adding Dependencies
import cv2
import numpy as np

# List of landmark connections
HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Wrist to thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Wrist to index
    (5, 9), (9, 10), (10, 11), (11, 12), # Index joint to Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Middle joint to Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Wrist to pinky
    (13, 17) # Pinky joint to Ring joint
]

def get_landmark_coords(result, frame):
    h, w, _ = frame.shape  # Get pixel dimensions of the camera feed
    hands_coords = []
    for hand_landmarks in result.hand_landmarks:
        coord_landmarks = []
        # Gets landmark coords
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            coord_landmarks.append((cx,cy))
        hands_coords.append(coord_landmarks)
    return hands_coords

# Returns handedness for detected hands
def handedness(result):
    detected = ["", ""]
    for i in range(len(result.hand_landmarks)):
        if result.handedness[i][0].category_name == "Left":
            detected[0] = "detected"
        if result.handedness[i][0].category_name == "Right":
            detected[1] = "detected"
    return detected

# Draws lines between each landmark based
def draw_hand_struct(result, frame):
    if result.hand_landmarks:
        coord_landmarks = get_landmark_coords(result, frame)
        for coords in coord_landmarks:
            for line in HAND_LINES:
                list_start, list_end = line
                point_start = coords[list_start]
                point_end = coords[list_end]
    
                cv2.line(frame, point_end, point_start, (203, 242, 172), 2)
                cv2.circle(frame, coords[list_end], 5, (191, 64, 191), cv2.FILLED)

# Returns detected gesture as text
def detect_gesture(result):
    if result.hand_landmarks:
        gesture = ""
        for i in range(len(result.hand_landmarks)):
            if result.gestures[i][0].category_name == "Thumb_Up":
                gesture = "Thumbs up"
            if result.gestures[i][0].category_name == "Open_Palm":
                gesture = "Open palm"
            if result.gestures[i][0].category_name == "Pointing_Up":
                gesture = "Point"
            if result.gestures[i][0].category_name == "Thumb_Down":
                gesture = "Thumbs Down"
            if result.gestures[i][0].category_name == "Closed_Fist":
                gesture = "Closed Fist"
            if result.gestures[i][0].category_name == "Victory":
                gesture = "Victory"
            if result.gestures[i][0].category_name == "ILoveYou":
                gesture = "I Love You"
        return gesture
            
# Draws on screen based on gesture input
def point_free_draw(result, frame, canvas, prev_point):
    if result.hand_landmarks:
        gesture = detect_gesture(result)
        coords = get_landmark_coords(result, frame)
        index = coords[8]

        if gesture == "Point":
            if prev_point is not None:
                cv2.line(canvas, prev_point, index, (0, 255, 0), 3)
            else:
                cv2.circle(canvas, index, 3, (0, 255, 0), -1)
            prev_point = index
        elif gesture == "Thumbs Down":
            canvas = None # Deletes drawing
        else:
            prev_point = None 

    return canvas, prev_point