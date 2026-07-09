
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
reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_face.npy")
reference = np.load(reference_path) if os.path.exists(reference_path) else None
if reference is None:
    print("No reference_face.npy found, please save 5 references using the 't' key before comparing faces.")

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

def normalize_landmarks(points, left_eye_idx=33, right_eye_idx=263):
    """
    Translate landmarks so they're centered at the origin (using centroid),
    then scale them so face size / distance-from-camera doesn't matter.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    left_eye = centered[left_eye_idx]
    right_eye = centered[right_eye_idx]
    eye_distance = np.linalg.norm(right_eye - left_eye)

    if eye_distance == 0:
        eye_distance = 1e-6

    return centered / eye_distance

def get_eye_distance_px(face_landmarks, frame_width, frame_height, left_eye_idx=33, right_eye_idx=263):
    """Raw eye-corner distance in pixels, before any normalization."""
    left = face_landmarks[left_eye_idx]
    right = face_landmarks[right_eye_idx]
    left_px = np.array([left.x * frame_width, left.y * frame_height])
    right_px = np.array([right.x * frame_width, right.y * frame_height])
    return np.linalg.norm(right_px - left_px) #the linalg.norm function calculates the Euclidean distance between the two points

def average_landmarks(instances):
    """
    Average a list of normalized landmark arrays into one reference.
    All instances must have the same shape (478, 3).
    """
    if not instances:
        raise ValueError("No instances to average.")
    stacked = np.stack(instances, axis=0)  # 
    return stacked.mean(axis=0)            # 

def compare_landmarks(a, b):
    """
    Mean per-point Euclidean distance between two normalized landmark sets.
    Lower = more similar.
    """
    diffs = np.linalg.norm(a - b, axis=1)
    return diffs.mean()

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    average = None

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

        MIN_EYE_DIST = 100   #I think this is a good minimum distance for the eyes to be apart in pixels
        MAX_EYE_DIST = 135

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
            

            live_eye_dist = get_eye_distance_px(face_landmarks, frame_width, frame_height)
            if live_eye_dist < MIN_EYE_DIST:
                cv2.putText(frame, f"Too far from camera (eye dist: {live_eye_dist:.1f}px). Move closer.", (x_min, y_max + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif live_eye_dist > MAX_EYE_DIST:
                cv2.putText(frame, f"Too close to camera (eye dist: {live_eye_dist:.1f}px). Move back.", (x_min, y_max + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Good distance from camera (eye dist: {live_eye_dist:.1f}px).", (x_min, y_max + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #cv2.putText(frame, f"Eye dist: {live_eye_dist:.1f}px", (x_min, y_max + 25),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if reference is not None:
                raw_points = landmarks_to_array(face_landmarks)
                live_normalized = normalize_landmarks(raw_points)
                score = compare_landmarks(live_normalized, reference)

                THRESHOLD = 0.05  # Lower = more similar
                if score < THRESHOLD:
                    label = f"MATCH ({score:.3f})"
                    color = (0, 255, 0)
                else:
                    label = f"NO MATCH ({score:.3f})"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        
        #press q to quit
        cv2.imshow('Face Landmarker', frame)
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):
            break

        if key == ord('t'):
            if face_landmarker_result.face_landmarks:
                current_face = face_landmarker_result.face_landmarks[0]
                eye_dist_px = get_eye_distance_px(current_face, frame_width, frame_height)

                

                if MIN_EYE_DIST <= eye_dist_px <= MAX_EYE_DIST:
                    raw_points = landmarks_to_array(current_face)
                    normalized_points = normalize_landmarks(raw_points)
                    saved_instances.append(normalized_points)

                    average = average_landmarks(saved_instances)

                    print(f"Saved instance {len(saved_instances)} (eye dist: {eye_dist_px:.1f}px)")
                elif eye_dist_px < MIN_EYE_DIST:
                    print(f"Too far from camera (eye dist: {eye_dist_px:.1f}px). Move closer.")
                else:
                    print(f"Too close to camera (eye dist: {eye_dist_px:.1f}px). Move back.")
            else:
                print("No face detected to save.")

    cap.release()
    cv2.destroyAllWindows()

    if average is not None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_face.npy")
        np.save(save_path, average)
        print(f"Saved reference (averaged from {len(saved_instances)} instances) to {save_path}")
    else:
        print("No instances saved — nothing written to disk.")