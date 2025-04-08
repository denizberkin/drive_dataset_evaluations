import os
from PIL import Image
from glob import glob

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.float().unsqueeze(0) / 255.0


def get_loaders(data_dir: str, batch_size: int, num_workers: int = 4,
                train_transform=None, test_transform=None):
    train_img_dir = os.path.join(data_dir, "train/images")
    train_mask_dir = os.path.join(data_dir, "train/masks")
    test_img_dir = os.path.join(data_dir, "test/images")
    test_mask_dir = os.path.join(data_dir, "test/masks")
    
    # dataset
    train_dataset = DriveDataset(
        train_img_dir, 
        train_mask_dir, 
        transform=train_transform
        )
    
    test_dataset = DriveDataset(
        test_img_dir, 
        test_mask_dir, 
        transform=test_transform
        )
    
    # dataloaders with transforms
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
