import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.preprocessing import set_seed, get_retinal_transforms
from utils.dataset import get_loaders
from utils.optim import optimizer_setup
from utils.swin_res_unet import SwinResUNet
from utils.unet import UNet


if __name__ == "__main__":
    # paths
    EXP_BASE_DIR = "experiments"
    EXP_NAME = "vanilla_unet"  # further used setting output directory to save models and logs
    DATA_DIR = "data/"
    # hyperparameters
    BATCH_SIZE = 4
    LR = 1.e-4
    NUM_EPOCHS = 40
    
    set_seed(42)  # set seed for reproducibility

    os.makedirs(EXP_BASE_DIR, exist_ok=True)
    os.makedirs(EXP_NAME, exist_ok=True)
    
    transforms = get_retinal_transforms(is_train=True)
    test_transforms = get_retinal_transforms(is_train=False)
    dl_train, dl_test, ds_train, ds_test = get_loaders(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE, 
        train_transform=transforms, 
        test_transform=test_transforms
        )
    
    # Change model here
    model = SwinResUNet()
    optimizer_setup(model=model, num_epochs=NUM_EPOCHS)  # setup optimizer and scheduler
    
    
    
    