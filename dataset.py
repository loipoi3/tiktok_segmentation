from torch.utils.data import Dataset
import os
import cv2
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

        # Add an extra dimension to the mask tensor to make it (1, height, width)
        mask = mask.unsqueeze(0)

        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask)