"""
Evaluation utilities for segmentation models.
This module contains functions for validating and evaluating segmentation models,
used by both training and testing pipelines.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import compute_dice_per_class, compute_iou_per_class, compute_pixel_accuracy

@torch.inference_mode()
def validate_point_model(model, dataloader, device, amp, dim=256, n_classes=3):
    """
    Validates a point-based segmentation model on a dataset.
    
    Args:
        model: The point-based segmentation model
        dataloader: DataLoader with validation or test data
        device: Device to run validation on
        amp: Whether to use mixed precision
        dim: Image dimension
        n_classes: Number of segmentation classes
        
    Returns:
        dict: Dictionary with validation metrics (dice, iou, acc, dice_per_class, iou_per_class)
    """
    model.eval()
    num_batches = len(dataloader)
    
    total_dice = torch.zeros(n_classes, device=device)
    total_iou = torch.zeros(n_classes, device=device)
    total_acc = 0
    
    # Keep track of per-class metrics
    class_samples = torch.zeros(n_classes, device=device)
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Validation', unit='batch', leave=False):        
            image = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = batch['mask'].to(device=device, dtype=torch.long)
            point = batch['point'].to(device=device, dtype=torch.float32)
            class_idx = batch['class'].to(device)
            
            # Get original size from the mask
            original_size = mask_true.shape[-2:]
            
            # Forward pass with point input
            mask_pred = model(image, point)
            mask_pred = mask_pred.argmax(dim=1)  # Convert to class indices
            
            # Resize masks to original size
            mask_pred = F.interpolate(mask_pred.unsqueeze(1).float(), size=original_size, mode='nearest').long().squeeze(1)
            
            # Compute Dice Score, IoU, and Pixel Accuracy
            dice_scores = compute_dice_per_class(mask_pred, mask_true, n_classes=n_classes)
            iou_scores = compute_iou_per_class(mask_pred, mask_true, n_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(mask_pred, mask_true)
            
            # Accumulate results
            total_dice += dice_scores
            total_iou += iou_scores
            total_acc += pixel_acc
            
            # Count samples per class (for averaging)
            for i in range(n_classes):
                class_samples[i] += (class_idx == i).sum()
    
    # Compute mean metrics
    mean_dice = total_dice.mean().item() / num_batches
    mean_iou = total_iou.mean().item() / num_batches
    mean_acc = total_acc / num_batches
    
    # Return metrics in a dictionary
    return {
        'dice': mean_dice,
        'iou': mean_iou,
        'acc': mean_acc,
        'dice_per_class': total_dice / num_batches,
        'iou_per_class': total_iou / num_batches
    } 