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
SCREENSHOT_FOLDER = "hand_image_data/validation_folder/Middle_finger"
#CSV_FOLDER = "landmark_data/single_hand/training/middle_finger.csv" 

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

def init_csv_file(path):
    file_header = []
    for i in range(21):
        file_header.extend([f"X{i}", f"Y{i}", f"Z{i}"])
    file_header.append("Gesture")
    

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(file_header)

# --- Stores landmark data in csv file ---------------
def save_landmark_data(world_hands, gesture):
    """ 
        Currently only saves 1 hand landmarks.
        If wanting to add logic for 2 hands, will 
        change lator.
    """
    # Creates file
    if not os.path.exists(CSV_FOLDER):
        init_csv_file(CSV_FOLDER)

    # Saves current frames landmarks 
    data = []
    for hand in world_hands:
        for landmark in hand:
            data.extend([landmark.x, landmark.y, landmark.z])
        data.append(gesture)

    # Writes to csv
    with open(CSV_FOLDER, "a", newline="") as file:
        csv.writer(file).writerow(data)

       