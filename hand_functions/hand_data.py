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
SCREENSHOT_FOLDER = "hand_image_data"
CSV_FOLDER = "landmark_data"

# Screenshots a image of the hand structure
def screenshot_hand(frame, result):
    model_input = np.zeros_like(frame) # Creates a black canvas the size of screen
    hv.draw_hand_struct(result, model_input) # Draws hand structure

    # Creates file path and name
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"hand_struct_{time_stamp}.jpg"
    full_path = os.path.join(SCREENSHOT_FOLDER, file_name)
    
    cv2.imwrite(full_path, model_input) # Saves file
    return None

# --- Stores landmark data in csv file ---------------
def save_landmark_data(world_hands):
    # Saves current frames landmarks 
    data = []
    for h_index, hand in enumerate(world_hands):
        for landmark_index, landmark in enumerate(hand):
            data.append({"Hand" : h_index, "Landmark": landmark_index, "X":landmark.x, "Y": landmark.y, "Z": landmark.z})
    
    dict_field = [
        'Hand',
        'Landmark',
        'X',
        'Y',
        'Z'
    ]
    
    # Creates file path
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"landmark_data_{time_stamp}.csv"
    full_path = os.path.join(CSV_FOLDER, file_name)

    # Writes to csv
    with open(full_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=dict_field)
        writer.writeheader()
        writer.writerows(data)

       