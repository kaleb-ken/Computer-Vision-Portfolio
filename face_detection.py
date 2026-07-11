import cv2
import time
import os
import multiprocessing as mp_process
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_mesh_connections import FACEMESH_TESSELATION
import numpy as np
import face_recognition

#better path finding
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_encoding.npy")

MATCH_THRESHOLD = 0.4  # face_recognition's typical default; lower = stricter
DOWNSCALE = 0.25  # run face_recognition on a quarter-size frame for speed


def face_recognition_process(frame_queue, result_queue, reference_encoding):
    """
    Runs in a completely separate process (its own Python interpreter, its
    own GIL), so face_recognition's computation can never block the main
    process's video loop, regardless of whether dlib releases the GIL.
    Always grabs the newest available frame, dropping any it falls behind on.
    """
    while True:
        frame_to_process = None
        # drain the queue, keeping only the most recent frame
        while not frame_queue.empty():
            item = frame_queue.get()
            if item is None:  # sentinel value used to signal shutdown
                return
            frame_to_process = item

        if frame_to_process is None:
            time.sleep(0.01)
            continue

        small_frame = cv2.resize(frame_to_process, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        face_locations_small = face_recognition.face_locations(small_frame) #get the locations of the faces in the frame
        face_encodings = face_recognition.face_encodings(small_frame, face_locations_small) #get the encodings of the faces in the frame

        # scale the face locations back up to the original frame size
        scale = int(1 / DOWNSCALE)
        face_locations = [
            (top * scale, right * scale, bottom * scale, left * scale)
            for (top, right, bottom, left) in face_locations_small
        ]
        # for each face, compare it to the reference encoding and determine if it's a match
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            label, color = None, (255, 255, 255)
            if reference_encoding is not None:
                distance = face_recognition.face_distance([reference_encoding], encoding)[0]
                if distance < MATCH_THRESHOLD:
                    label, color = f"MATCH ({distance:.3f})", (0, 255, 0)
                else:
                    label, color = f"NO MATCH ({distance:.3f})", (0, 0, 255)

            # keep only the latest result in the queue too
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except Exception:
                    break
            result_queue.put((encoding, (top, right, bottom, left), label, color))

#main function that runs the face detection and recognition
def main():
    reference_encoding = np.load(reference_path) if os.path.exists(reference_path) else None
    if reference_encoding is None:
        print("No reference_encoding.npy found, please save some references using the 't' key before comparing faces.")

    #shortcut for mediapipe classes
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    #Creates a face landmarker object with the specified model and options
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)

    saved_encodings = []  # list of 128-number face_recognition encodings

    # --- set up the separate process and its communication queues ---
    frame_queue = mp_process.Queue(maxsize=1)
    result_queue = mp_process.Queue(maxsize=1)
    worker = mp_process.Process(
        target=face_recognition_process,
        args=(frame_queue, result_queue, reference_encoding),
        daemon=True
    )
    worker.start() #worker process that runs the face recognition in a separate process to avoid blocking the main thread

    # main loop: capture frames, run MediaPipe, and display results 
    current_encoding = None
    last_box = None
    last_label = None
    last_color = (255, 255, 255)


    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        average_encoding = None

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

            # MediaPipe: used only for drawing the mesh visuals 
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_height, frame_width = frame.shape[:2]

            # track MediaPipe's real-time box position for label placement 
            mediapipe_box = None
            for face_landmarks in face_landmarker_result.face_landmarks:
                x_coords = [landmark.x for landmark in face_landmarks]
                y_coords = [landmark.y for landmark in face_landmarks]

                x_min = int(min(x_coords) * frame_width)
                x_max = int(max(x_coords) * frame_width)
                y_min = int(min(y_coords) * frame_height)
                y_max = int(max(y_coords) * frame_height)
                mediapipe_box = (x_min, y_min, x_max, y_max)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                for start_idx, end_idx in FACEMESH_TESSELATION:
                    start = face_landmarks[start_idx]
                    end = face_landmarks[end_idx]
                    x1, y1 = int(start.x * frame_width), int(start.y * frame_height)
                    x2, y2 = int(end.x * frame_width), int(end.y * frame_height)
                    cv2.line(frame, (x1, y1), (x2, y2), (231, 225, 93), 1)

            # --- hand the latest frame to the worker process (non-blocking) 
            if frame_queue.empty():
                try:
                    frame_queue.put_nowait(rgb_frame)
                except Exception:
                    pass

            # pick up the latest result if one is available (non-blocking)
            if not result_queue.empty():
                try:
                    current_encoding, last_box, last_label, last_color = result_queue.get_nowait()
                except Exception:
                    pass

            if mediapipe_box is not None and last_label is not None:
                x_min, y_min, x_max, y_max = mediapipe_box
                cv2.putText(frame, last_label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)

            #press q to quit
            cv2.imshow('Face Landmarker', frame)
            key = cv2.waitKey(5) & 0xFF

            if key == ord('q'):
                break

            if key == ord('t'):
                if current_encoding is not None:
                    saved_encodings.append(current_encoding)
                    average_encoding = np.mean(saved_encodings, axis=0)
                    print(f"Saved instance {len(saved_encodings)}")
                else:
                    print("No face detected to save.")

        cap.release()
        cv2.destroyAllWindows()

        # signal the worker process to stop, then wait for it to exit cleanly
        frame_queue.put(None)
        worker.join(timeout=2)

        if average_encoding is not None:
            np.save(reference_path, average_encoding)
            print(f"Saved reference (averaged from {len(saved_encodings)} instances) to {reference_path}")
        else:
            print("No instances saved — nothing written to disk.")


if __name__ == "__main__":
    main()