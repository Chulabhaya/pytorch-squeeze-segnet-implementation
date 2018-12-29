from squeeze_segnet import SqueezeSegNet, SqueezeSegNetOptimized
from dataset import CustomSegNetDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from pathlib import Path
plt.switch_backend('agg')
plt.axis('off')

# Training parameters
IN_CHANNELS = 3
OUT_CLASSES = 2
DATASET = 'Custom'

# Check for and enable GPU usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up input/output file paths
project_root = Path(__file__).parent 
image_list = project_root / 'data' / DATASET / 'val.txt'
image_dir = project_root / 'data' / DATASET / 'val/'
mask_dir = project_root / 'data' / DATASET / 'valannot/'
model_save_dir = project_root / 'models/' 
prediction_save_dir = project_root / 'predictions/' 

# Load prediction dataset
val_dataset = CustomSegNetDataset(image_list, image_dir, mask_dir, OUT_CLASSES)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# Initialize model
model = SqueezeSegNetOptimized(IN_CHANNELS, OUT_CLASSES)
model = model.to(device)

# Set up loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()

# Load previously trained models
model.load_state_dict(torch.load(model_save_dir / "model_trained.pth"))

# Prediction function
def predict_model(val_dataloader, model, criterion):
    model.train()

    for batch_idx, (image, mask) in enumerate(val_dataloader):
        image = image.to(device)
        mask = mask.to(device)

        # Run current batch of input images through network 
        prediction = model(image)
        loss = criterion(prediction, mask)

        # Save plots of predictions vs. inputs and masks
        for idx, prediction in enumerate(prediction):
            target_mask = mask[idx]
            input_image = image[idx]

            fig = plt.figure()

            a = fig.add_subplot(1,3,1)
            plt.imshow(input_image.transpose(0, 2))
            a.set_title('Input Image')

            a = fig.add_subplot(1,3,2)
            predicted_mx = prediction.detach().cpu().numpy()
            predicted_mx = predicted_mx.argmax(axis=0)
            plt.imshow(predicted_mx)
            a.set_title('Predicted Mask')

            a = fig.add_subplot(1,3,3)
            target_mx = target_mask.detach().cpu().numpy()
            plt.imshow(target_mx)
            a.set_title('Ground Truth')

            fig.savefig(prediction_save_dir / "prediction_{}_{}.png".format(batch_idx, idx))

            plt.close(fig)

# Call prediction function and make predictions with model on input data
predict_model(val_dataloader, model, criterion)
