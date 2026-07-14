import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt #for data viz
import pandas as pd 
import numpy as np


#Definding the dataset---------------------
class HandSignDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes #returns the data classes from the image folder
    

data_dir = "./data"
dataset = HandSignDataset(data_dir) #NEED TO GET DATA FAHHH

target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()} #dictionary that links each number with a correct label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

#batching the dataset now

Dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#pytorch model-----------------------------------







