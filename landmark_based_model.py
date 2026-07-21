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


# --- Setting up Neural Net -----------------
class Model(nn.Module):
    def __init__(self, input_layer=63, h1=16, h2=16, output=2):
        """ Initialising the model
        Args:
            input_layer: data input
            h1: Dense layer
            h2: Dense layer
            output: Confidence value for 1 of 10 outputs
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
CSV_FOLDER = "landmark_data/single_hand/training/middle_finger.csv" 

class LandmarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = pd.read_csv(data_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

   

training_data = LandmarkDataset(CSV_FOLDER)

training_load = DataLoader(
    dataset=training_data,
    batch_size=32,
    shuffle=True,
)

# --- Training Loop -----------------
# Setting up loop
model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 10
losses = []

for epoch in range(NUM_EPOCHS):
    Model.train()
    running_loss = 0.0

    for batch_index, gestures in enumerate(training_load):

