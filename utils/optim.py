import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def optimizer_setup(model: nn.Module, num_epochs: int = 40, 
                    lr: float = 4.e-4) -> tuple:
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    return optimizer, scheduler
