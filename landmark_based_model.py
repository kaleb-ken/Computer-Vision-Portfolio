"""
gesture_model.py
========================
Trains a MLP model for gesture detection. 
Uses landmark data for training.
"""
# --- Importing dependencies ---------------
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


# --- Setting up Neural Net -----------------
class Model(nn.Module):
    def __init__(self, input_layer=63, h1=16, h2=16, output=2):
        """ Initialising the model
        Args:
            input_layer: data input
            h1: Dense layer
            h2: Dense layer
            output: Gesture detected or not
        """
        super().__init__()
        self.conns1 = nn.Linear(input_layer, h1)
        self.conns2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)
    
    def forward(self, x):
        """ Connects each layer together
        Args:
            x: Data batch
        """
        x = f.relu(self.conns1(x))
        x = f.relu(self.conns2(x))
        x = self.out(x)

        return x
    
# --- Setting up Dataset -----------------
CSV_TRAIN = "landmark_data/single_hand/training/middle_finger_train.csv" 
CSV_TEST = "landmark_data/single_hand/testing/middle_finger_test.csv"

class LandmarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = pd.read_csv(data_dir)
        self.transform = transform

        self.classes = sorted(self.data.iloc[:, -1].unique())
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        landmarks = torch.tensor(row.iloc[:-1].values.astype("float32"))
        label_str = row.iloc[-1]
        gesture = torch.tensor(self.class_to_index[label_str], dtype=torch.long)
        return landmarks, gesture

   


training_data = LandmarkDataset(CSV_TRAIN)
testing_data = LandmarkDataset(CSV_TEST)

training_load = DataLoader(
    dataset=training_data,
    batch_size=32,
    shuffle=True
)
testing_load = DataLoader(
    dataset=testing_data,
    batch_size=32,
    shuffle=False
)

# --- Training Loop -----------------
# Setting up loop
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 10
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(training_load, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for landmarks, gestures in progress_bar:
        
        optimizer.zero_grad()
        output = model(landmarks)
        loss = criterion(output, gestures)
        loss.backward() #back propagation
        optimizer.step()
        running_loss += loss.item() * landmarks.size(0)
        progress_bar.set_postfix({'loss': loss.item()})  # Update the progress bar with the current loss
    
    epoch_loss = running_loss / len(training_load.dataset)
    losses.append(epoch_loss)

# --- Testing Loop -----------------------

model.eval()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for landmarks, gestures in testing_load:
        output = model(landmarks)
        loss = criterion(output, gestures)
        test_loss += loss.item() * landmarks.size(0)

        predictions = torch.argmax(output, dim=1)  
        correct += (predictions == gestures).sum().item()
        total += gestures.size(0)
avg_test_loss = test_loss / len(testing_load.dataset)
accuracy = correct / total

print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {accuracy:.2%}")

