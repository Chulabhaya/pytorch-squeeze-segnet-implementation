from squeeze_segnet import SqueezeSegNet, SqueezeSegNetOptimized
from dataset import CustomSegNetDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from pathlib import Path
from utility import calculateIoU

# Training parameters
IN_CHANNELS = 3
OUT_CLASSES = 2
LEARNING_RATE = 0.0001
DATASET = 'Custom'

# Check for and enable GPU usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up input/output file paths
project_root = Path(__file__).parent 
image_list = project_root / 'data' / DATASET / 'train.txt'
image_dir = project_root / 'data' / DATASET / 'train/'
mask_dir = project_root / 'data' / DATASET / 'trainannot/'
model_save_dir = project_root / 'models/' 

# Load training dataset
train_dataset = CustomSegNetDataset(image_list, image_dir, mask_dir, OUT_CLASSES)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model
model = SqueezeSegNetOptimized(IN_CHANNELS, OUT_CLASSES)
model = model.to(device)

# Set up loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
model_parameters = model.parameters()
optimizer = torch.optim.Adam(model_parameters, lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

# Training function
def train_model(train_dataloader, model, criterion, optimizer, scheduler, num_epochs=100):
    model.train()

    # Initialize and keep track of loss to save models with least loss
    prev_epoch_loss = float('inf')

    for epoch in range(num_epochs):
        current_epoch_loss = 0
        current_epoch_iou = 0
        scheduler.step()

        for image, mask in train_dataloader:
            image = image.to(device)
            mask = mask.to(device)

            # Run current batch of input images through network
            prediction = model(image)

            # Calculate IoU of predictions vs. masks
            prediction_temp = prediction.detach()
            mask_temp = mask.detach()
            iou_score = calculateIoU(prediction_temp, mask_temp)

            # Update parameters
            optimizer.zero_grad()
            loss = criterion(prediction, mask)
            loss.backward()
            optimizer.step()

            current_epoch_loss += loss.float()
            current_epoch_iou += iou_score

        current_epoch_iou /= len(train_dataloader)
        print("Epoch {}, Loss: {}, Mean IoU: {}".format(epoch, current_epoch_loss, current_epoch_iou))

        if current_epoch_loss < prev_epoch_loss:
            prev_epoch_loss = current_epoch_loss
            torch.save(model.state_dict(), model_save_dir / "model_trained.pth")

# Call training function and train model
train_model(train_dataloader, model, criterion, optimizer, scheduler, 100)

