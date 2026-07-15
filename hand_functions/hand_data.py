""""
hand_data.py
=================
Takes screenshot of hand structure and 
saves landmark data for gesture training.
"""
# Adding dependencies
import numpy as np
import cv2
import hand_functions.hand_visuals as hv

#
def screenshot_hand(frame, result):
    hv.draw_hand_struct(result, frame)
    cv2.imwrite("hand_struct.jpg", frame)

