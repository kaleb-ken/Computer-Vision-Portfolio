""""
hand_data.py
=================
Takes screenshot of hand structure and 
saves landmark data for gesture training.
"""
# Adding dependencies
import datetime
import numpy as np
import os
import csv
import cv2
import hand_functions.hand_visuals as hv

# --- CHANGE FILE PATH WHEN CREATIN NEW DATASETS ------------------
SCREENSHOT_FOLDER = "hand_image_data/test_folder/Middle_finger"
CSV_FOLDER = "landmark_data/middle_finger_test.csv" 
SCREENSHOT_FOLDER = "hand_image_data/train_folder/Shadow_clone"
#CSV_FOLDER = "landmark_data/single_hand/testing/middle_finger_test.csv" 


# --- Screenshots a image of the hand structure ---------------
def screenshot_hand(frame, result):
    model_input = np.zeros_like(frame) # Creates a black canvas the size of screen
    hv.draw_hand_struct(result, model_input) # Draws hand structure

    # Creates file path and name
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"hand_struct_{time_stamp}.jpg"
    full_path = os.path.join(SCREENSHOT_FOLDER, file_name)
    
    cv2.imwrite(full_path, model_input) # Saves file
    return None

# --- Creates csv file with header ----------------
def init_csv_file(path):
    file_header = []
    hand = "L"
    for i in range(2):
        for i in range(21):
            file_header.extend([f"{hand}x{i}", f"{hand}y{i}", f"{hand}z{i}"])
        hand = "R"
    file_header.append("Gesture")
    

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(file_header)

# --- Stores landmark data in csv file ---------------
def save_landmark_data(result, gesture):
    """ Stores landmark data to csv
        Args:
            result: hand detections
            gesture: Label for dataset
            
        Saves data for left and/or right, sets hand 
        data to zero if not detected.
    """
    # Creates file
    if not os.path.exists(CSV_FOLDER):
        init_csv_file(CSV_FOLDER)

    # Setting up variables
    csv_input = []
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    # Saves current frames landmarks 
    if result.hand_world_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_world_landmarks, result.handedness):
            data = []
            for landmark in landmarks:
                data.extend([landmark.x, landmark.y, landmark.z])
            if  handedness[0].category_name == "Left":
                left_hand = data
            elif handedness[0].category_name == "Right":
                right_hand = data

    # Setting up list for csv
    csv_input.extend(left_hand)
    csv_input.extend(right_hand)
    csv_input.append(gesture)
    
    # Writes to csv
    with open(CSV_FOLDER, "a", newline="") as file:
        csv.writer(file).writerow(csv_input)

       