from xml.parsers.expat import model

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
from tqdm.notebook import tqdm


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

target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()} #dictionary that links each number with a correct label

transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
])

dataset = HandSignDataset(data_dir, transform=transform) #NEED TO GET DATA FAHHH

#batching the dataset now

Dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#pytorch model-----------------------------------

class SimpleHandClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleHandClassifier, self).__init__()
        self.basemodel = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0) #num_classes=0 strips the head, returns pooled features

        enet_out_size = self.basemodel.num_features

        #make a classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)
    
    def forward(self, x):

        x = self.basemodel(x)
        output = self.classifier(x)
        return output
    
#Training loop ---------------------------

train_folder = "./data/train"
val_folder = "./data/val"
test_folder = "./data/test"

train_dataset = HandSignDataset(train_folder, transform=transform)
val_dataset = HandSignDataset(val_folder, transform=transform)
test_dataset = HandSignDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#so we have our datasets and our loop here is the real training loop.

NUM_EPOCHS = 5 #CAN BE CHANGED TO WHATEVERrrrrrrrrrrrrr r r nfjenqfj
train_loss, val_losses = [], []

model = SimpleHandClassifier(num_classes=len(dataset.classes)) #should be the number of classes in the dataset, eg num_classes = 3
#model.to(device) #if using GPU, uncomment this line

#Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #learnign rate can be changed to whatever you want, 0.001 is a good starting point

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        #inputs, labels = inputs.to(torch.device), labels.to(torch.device) #if using GPU, uncomment this line


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) #calculating loss
        loss.backward() #back propagation
        optimizer.step() 
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss.append(epoch_loss)

    # Validation
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
    
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_epoch_loss)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')