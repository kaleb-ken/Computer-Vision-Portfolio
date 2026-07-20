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

# --- Setting up Neural Net -----------------
class Model(nn.Module):
    def __init__(self, input_layer=63, h1=16, h2=16, output=10):
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