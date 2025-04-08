import random

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# set seed for reproducibility, call this before any processing
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_retinal_transforms(is_train: bool = True) -> A.Compose:
    if is_train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        mask_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=1.0),
            ToTensorV2()
        ])
            
    else:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        mask_transform = A.Compose([
            ToTensorV2()
        ])
    
    return transform, mask_transform
        
        
        