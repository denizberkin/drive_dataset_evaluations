import os

import torch
import pandas as pd

from utils.preprocessing import set_seed, get_retinal_transforms
from utils.dataset import get_loaders
from utils.optim import optimizer_setup
from utils.loss import BCEDiceLoss
from src.train import train_model

from utils.swin_res_unet import SwinResUNet
from utils.swin_unet import SwinUnet
from utils.unet import UNet


if __name__ == "__main__":
    # paths
    EXP_BASE_DIR = "experiments"
    EXP_NAME = "unet_aug_scheduler"  # further used setting output directory to save models and logs
    DATA_DIR = "data/"
    # hyperparameters
    IMG_SIZE = 512
    BATCH_SIZE = 4
    LR = 4.e-4
    NUM_EPOCHS = 40
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    set_seed(42)  # set seed for reproducibility

    os.makedirs(EXP_BASE_DIR, exist_ok=True)
    os.makedirs(EXP_NAME, exist_ok=True)
    
    transforms = get_retinal_transforms(resize_to=IMG_SIZE, is_train=True)
    test_transforms = get_retinal_transforms(resize_to=IMG_SIZE, is_train=False)
    dl_train, dl_test, ds_train, ds_test = get_loaders(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE, 
        train_transform=transforms, 
        test_transform=test_transforms
        )
    
    # Change model here
    model = SwinUnet().to(DEVICE)
    optimizer, scheduler = optimizer_setup(model=model, num_epochs=NUM_EPOCHS, lr=LR)
    criterion = BCEDiceLoss().to(DEVICE)
    
    # train the model
    model, val_results = train_model(
        model=model,
        train_loader=dl_train,
        val_loader=dl_test,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
        )
    
    # Save the model
    model_save_path = os.path.join(EXP_BASE_DIR, EXP_NAME, f"{str(model)}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save the training results
    results_save_path = os.path.join(EXP_BASE_DIR, EXP_NAME, f"{str(model)}_results.csv")
    val_results_df = pd.DataFrame(val_results)  # key, list of values
    val_results_df.to_csv(results_save_path, index=False)
    print(f"Val results saved to {results_save_path}")
    