import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CustomSegNetDataset(Dataset):
    """
    Custom Dataset used for SegNet. 
    """
    def __init__(self, image_list, image_dir, mask_dir, num_classes, transform=None):
        """
        Args:
            image_list (string): Directory to text file containing image names (images and masks share same file names)
            image_dir (string): Directory with images. 
            mask_dir (string): Directory with image masks. 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = open(image_list, "rt").read().split("\n")[:-1]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = self.image_dir / image_name
        mask_path = self.mask_dir / image_name

        image = torch.FloatTensor(self.load_image(image_path))
        mask = torch.LongTensor(self.load_mask(mask_path))
        return image, mask

    def load_image(self, path=None):
        raw_image = Image.open(path).convert('RGB')
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0
        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        imx_t[imx_t==255] = self.num_classes
        return imx_t

if __name__ == "__main__":
    project_root = Path(__file__).parent 
    image_list = project_root / 'data' / 'Custom' / 'train.txt'
    image_dir = project_root / 'data' / 'Custom' / 'train/'
    mask_dir = project_root / 'data' / 'Custom' / 'trainannot/'

    objects_dataset = CustomSegNetDataset(image_list, image_dir, mask_dir)
    image, mask = objects_dataset[0]
    image.transpose_(0, 2)
    fig = plt.figure()

    a = fig.add_subplot(1,2,1)
    plt.imshow(image)

    a = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    plt.show()