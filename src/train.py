import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from src.test import validate_one_epoch


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device
    ):
    model.train()
    epoch_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        preds = model(images)
        loss = criterion(preds, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    scheduler: LRScheduler = None,
) -> tuple:
    for epoch in range(num_epochs):
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")
        tqdm.write("-" * 30)

        # train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        tqdm.write(f"Training Loss: {train_loss:.4f}")

        # validate for one epoch
        val_loss, val_mean_metrics, val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        tqdm.write(f"Validation Loss: {val_loss:.4f}")
        tqdm.write("Validation Metrics:")
        for metric_name, metric_value in val_mean_metrics.items():
            tqdm.write(f"  {metric_name}: {metric_value:.4f}", end="\t")

        if scheduler:
            scheduler.step()
            
    return model, val_metrics