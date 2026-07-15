""""
hand_data.py
=================
Takes screenshot of hand structure and 
saves landmark data for gesture training.
"""
# Adding dependencies
import datetime
import os
import cv2
import hand_functions.hand_visuals as hv
screenshot_folder = "hand_image_data"

# Screenshots a image of the hand structure
def screenshot_hand(frame, result):
    hv.draw_hand_struct(result, frame)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"hand_struct_{time_stamp}.jpg"
    full_path = os.path.join(screenshot_folder, file_name)
    
    cv2.imwrite(full_path, frame)
    return None
