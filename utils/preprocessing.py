import random

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# set seed for reproducibility, call this before any processing
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_retinal_transforms(resize_to: int = 512, is_train: bool = True) -> A.Compose:
    if is_train:
        transform = A.Compose([
            A.Resize(height=resize_to, width=resize_to),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=1.0),
            A.Normalize(mean=(0.5), std=(0.5)),
            ToTensorV2()
        ])
        
        mask_transform = A.Compose([
            A.Resize(height=resize_to, width=resize_to),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=1.0),
            ToTensorV2()
        ])
            
    else:
        transform = A.Compose([
            A.Resize(height=resize_to, width=resize_to),
            A.Normalize(mean=(0.5), std=(0.5)),
            ToTensorV2()
        ])
        
        mask_transform = A.Compose([
            A.Resize(height=resize_to, width=resize_to),
            ToTensorV2()
        ])
    
    return transform, mask_transform
        
        
def resize_to_original(pred: torch.Tensor, original_size: tuple) -> torch.Tensor:
    """
    pred (torch.Tensor): (B, C, H, W)
    original_size (tuple): (H, W)
    """
    return F.interpolate(
        pred, 
        size=original_size, 
        mode='bilinear', 
        align_corners=True
    )
    
    
class ResizeProcessor:
    """ to maybe use in unit tests """
    def __init__(self, target_size=512):
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> tuple:
        original_size = (image.shape[0], image.shape[1])
        
        resized = cv2.resize(image, (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_LINEAR)
        
        tensor = torch.from_numpy(resized / 255.0, np.float32).permute(2, 0, 1).unsqueeze(0)
        
        return tensor, original_size
        
    def postprocess(self, prediction: torch.Tensor, original_size: tuple) -> np.ndarray:
        resized = resize_to_original(prediction, original_size)
        np_pred = resized.squeeze().cpu().numpy()
        return (np_pred * 255).astype(np.uint8)