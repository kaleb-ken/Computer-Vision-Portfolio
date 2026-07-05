
import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS, FACEMESH_IRISES
import numpy as np
import matplotlib.pyplot as plt

#better path finding
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")

#shortcut for mediapipe classes
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#Creates a face landmarker object with the specified model and options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

saved_instances = []  # list of numpy arrays, one per saved snapshot

def landmarks_to_array(face_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)

        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        #print(face_landmarker_result)  # prints out data for each detected face, including landmarks and bounding boxes

        frame_height, frame_width = frame.shape[:2]

        # Loop over each detected face (usually just one, but supports more)
        for face_landmarks in face_landmarker_result.face_landmarks:
            # Pull out all x and y coords for this face
            x_coords = [landmark.x for landmark in face_landmarks]
            y_coords = [landmark.y for landmark in face_landmarks]

            # Draw each landmark as a small circle on the frame
            #for x, y in zip(x_coords, y_coords):
                #px = int(x * frame_width)
                #py = int(y * frame_height)
                #cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            # Convert normalized (0-1) coords to actual pixel coords
            x_min = int(min(x_coords) * frame_width)
            x_max = int(max(x_coords) * frame_width)
            y_min = int(min(y_coords) * frame_height)
            y_max = int(max(y_coords) * frame_height)

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            for start_idx, end_idx in FACEMESH_TESSELATION:
                start = face_landmarks[start_idx]
                end = face_landmarks[end_idx]

                x1 = int(start.x * frame_width)
                y1 = int(start.y * frame_height)
                x2 = int(end.x * frame_width)
                y2 = int(end.y * frame_height)

                cv2.line(frame, (x1, y1), (x2, y2), (231, 225, 93), 1)
        

        #press q to quit
        cv2.imshow('Face Landmarker', frame)
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):
            break

        if key == ord('t'):
            if face_landmarker_result.face_landmarks:
                # Convert landmarks to numpy array and save
                landmarks_array = landmarks_to_array(face_landmarker_result.face_landmarks[0]) #currently only saving the first detected face
                saved_instances.append(landmarks_array)
                print(f"Saved instance {len(saved_instances)}")
            else:
                print("No face detected to save.")




    cap.release()
    cv2.destroyAllWindows()