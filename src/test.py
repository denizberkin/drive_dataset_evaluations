import torch
from tqdm import tqdm
from utils.metrics import METRICS_DICT


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    metrics = {key: [] for key in METRICS_DICT.keys()}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)
            epoch_loss += loss.item()

            for metric_name, metric_fn in METRICS_DICT.items():
                metrics[metric_name].append(metric_fn(preds, masks).item())

    mean_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
    return epoch_loss / len(dataloader), mean_metrics, metrics