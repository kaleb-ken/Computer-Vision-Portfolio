"""
kalman_filter.py
====================
Helps optimise hand detection code
"""
# Adding libraries
import cv2
import numpy as np


def kalman():

    kalman_filter = cv2.KalmanFilter(4,2)
    kalman_filter.transitionMatrix = np.array(
        [[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0],[0, 0, 0, 1]],
        dtype=np.float32
    )
    kalman_filter.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], 
        dtype=np.float32
    )
    kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman_filter.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)
    kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-3

    return kalman_filter