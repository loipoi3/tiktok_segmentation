from torch.utils.data import Dataset
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np


class TikTokDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, 'images')))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir + '/images', os.listdir(os.path.join(self.root_dir, 'images'))[idx])
        mask_path = os.path.join(self.root_dir + '/masks', os.listdir(os.path.join(self.root_dir, 'masks'))[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask[mask == 255.0] = 1
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image'].permute(1, 2, 0)
            mask = augmentations['mask']

        return torch.tensor(image), torch.tensor(mask)


import matplotlib.pyplot as plt

TRANSFORM_TRAIN = A.Compose([A.HorizontalFlip(p=0.5),
                             A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                             A.RandomBrightnessContrast(p=0.5),
                             A.GaussianBlur(p=0.5),
                             A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                             A.ChannelShuffle(p=0.5),
                             ,
                             ToTensorV2()])
a = TikTokDataset(root_dir='./data/train', transform=TRANSFORM_TRAIN)

# Get a sample from the dataset
sample_idx = 0
image, mask = a[sample_idx]

# Convert the mask to binary
mask = (mask > 0.5).to(torch.uint8) * 255

# Convert image back to PyTorch tensor
image = torch.tensor(image)

# Plot the image and mask
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)  # Transpose image from (C, H, W) to (H, W, C)
axes[0].set_title("Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Mask")
axes[1].axis("off")

plt.show()