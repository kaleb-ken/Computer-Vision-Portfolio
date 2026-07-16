"""
gesture_model.py
========================
Trains a CNN model for gesture detection. 
Uses landmark data for training.
"""
# Importing dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms