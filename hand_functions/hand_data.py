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
import cv2
import hand_functions.hand_visuals as hv
screenshot_folder = "hand_image_data"

# Screenshots a image of the hand structure
def screenshot_hand(frame, result):
    model_input = np.zeros_like(frame) # Creates a black canvas the size of screen
    hv.draw_hand_struct(result, model_input) # Draws hand structure

    # Creates file path and name
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"hand_struct_{time_stamp}.jpg"
    full_path = os.path.join(screenshot_folder, file_name)
    
    cv2.imwrite(full_path, model_input) # Saves file
    return None

# Stores landmark data in csv file
def save_landmark_data(result):
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data = []
    for landmark in result.hand_landmarks:
        for coords in landmark:
            data.append({"landmark":"1", "X":coords.x, "Y": coords.y, "Z": coords.z})