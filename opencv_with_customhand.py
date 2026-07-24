"""
opencv_with_customhand.py
======================
Live inference using the trained image-based hand sign classifier
"""

import time
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import json
from mediapipe.tasks.python import vision
from PIL import Image
import torchvision.transforms as transforms
import hand_functions.hand_visuals as hv
import hand_functions.one_euro as OE #we love one euro


# Load class mapping

with open("class_mapping.json", "r") as f:
    target_to_class = json.load(f)
    target_to_class = {int(k): v for k, v in target_to_class.items()}

num_classes = len(target_to_class)


# Model definition, matches the training script. Ts a simple CNN for image classification.

class SimpleHandClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleHandClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


# Load trained weights

model = SimpleHandClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("handsign_model.pth", map_location="cpu")) 
model.eval()


# Preprocessing, matches eval_transform from training (no augmentation)

transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
])

# ---------------------------------------------------
# Setting up capture

model_input = None
feed = cv2.VideoCapture(0)
feed.set(cv2.CAP_PROP_FRAME_HEIGHT, value=500)
feed.set(cv2.CAP_PROP_FRAME_WIDTH, value=500)


# Setting up hand detection (same as hand_detection.py)

model_path = "./models/hand_landmarker.task"
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)
start_time = time.time()

# ---------------------------------------------------
# Live video loop

while True:
    ret, frame = feed.read() #take a frame from the webcam
    if not ret:
        break
    if model_input is None:
        model_input = np.zeros_like(frame) # Creates a black canvas the size of screen
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    result = OE.optimise_landmarks(result)

    hv.draw_hand_struct(result, frame)
    hv.draw_hand_struct(result, model_input)

    predicted_label = "No hand detected"
    confidence_text = ""

    if result.hand_landmarks:
        # convert skeleton canvas (BGR) to RGB for the model, same as training
        skeleton_rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(skeleton_rgb)
        input_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad(): #load the model and make predictions
            output = model(input_tensor) #RUNS INFERENCE HERE!!!
            probs = torch.softmax(output, dim=1) #apply softmax to get probabilities
            all_labels = [f"{target_to_class[i]}: {probs[0][i]:.3f}" for i in range(num_classes)]
            print(all_labels)
            confidence, predicted_idx = torch.max(probs, dim=1) #get the highest probability and its index
            predicted_label = target_to_class[predicted_idx.item()]
            confidence_text = f"{confidence.item()*100:.1f}%"

        # find hand's bottom point (frame coords) to position the label under it
        landmarks = result.hand_landmarks[0]
        frame_height, frame_width, _ = frame.shape
        x_coords = [lm.x * frame_width for lm in landmarks]
        y_coords = [lm.y * frame_height for lm in landmarks]
        hand_bottom = int(max(y_coords))
        hand_center_x = int(sum(x_coords) / len(x_coords))

        label_y = min(hand_bottom + 30, frame_height - 40)
        confidence_y = label_y + 30

        text_size = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        label_x = max(hand_center_x - text_size[0] // 2, 10)

        cv2.putText(frame, predicted_label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, confidence_text, (label_x, confidence_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('capture', frame)
    cv2.imshow('Model Input', model_input)
    model_input = None

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()