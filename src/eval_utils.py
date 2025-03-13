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
def validate_point_model(net, dataloader, device, amp, n_classes=3):
    """
    Validates a point-based segmentation model on a dataset.
    
    Args:
        net (torch.nn.Module): The model to validate
        dataloader (torch.utils.data.DataLoader): The dataloader for validation data
        device (torch.device): The device to run validation on
        amp (bool): Whether to use automatic mixed precision
        n_classes (int): Number of classes for segmentation
        
    Returns:
        tuple: A tuple containing (dice score, iou score, pixel accuracy, dice per class, iou per class)
    """
    net.eval()
    num_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    pixel_acc = 0
    dice_per_class = torch.zeros(n_classes, device=device)
    iou_per_class = torch.zeros(n_classes, device=device)
    
    with torch.inference_mode():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_batches, desc="Validation round", unit="batch", leave=False):
                images = batch['image']
                true_masks = batch['mask']
                point_heatmap = batch['point']  # This is already a heatmap, not coordinates
                
                # Move to device
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                point_heatmap = point_heatmap.to(device=device, dtype=torch.float32)
                
                # Forward pass - only passing image and point heatmap as expected by the model
                masks_pred = net(images, point_heatmap)
                
                # Get predictions
                if n_classes == 1:
                    masks_pred = (torch.sigmoid(masks_pred) > 0.5).float()
                else:
                    masks_pred = F.one_hot(masks_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                    
                # Create a mask to identify void pixels (255)
                void_pixels = true_masks == 255
                
                # For metrics calculation, create a mask that excludes void pixels
                metrics_mask = true_masks.clone()
                
                # Convert ground truth to one-hot encoding
                if n_classes > 1:
                    # Temporarily set void pixels to a valid class index to avoid errors in one_hot
                    metrics_mask[void_pixels] = 0
                    true_masks_one_hot = F.one_hot(metrics_mask, n_classes).permute(0, 3, 1, 2).float()
                    
                    # Create a mask for valid pixels (not void)
                    valid_mask = ~void_pixels
                    
                    # Compute metrics only on valid pixels
                    for c in range(n_classes):
                        pred_c = masks_pred[:, c]
                        true_c = true_masks_one_hot[:, c]
                        
                        # Apply the valid mask
                        pred_c_valid = pred_c * valid_mask.float()
                        true_c_valid = true_c * valid_mask.float()
                        
                        # Calculate intersection and union for valid pixels
                        intersection = (pred_c_valid * true_c_valid).sum()
                        pred_sum = pred_c_valid.sum()
                        true_sum = true_c_valid.sum()
                        union = pred_sum + true_sum - intersection
                        
                        # Calculate Dice score and IoU for this class
                        if (pred_sum + true_sum) > 0:
                            dice_per_class[c] += (2.0 * intersection) / (pred_sum + true_sum)
                        if union > 0:
                            iou_per_class[c] += intersection / union
                    
                    # Calculate overall pixel accuracy excluding void pixels
                    masks_pred_cls = masks_pred.argmax(dim=1)
                    correct = (metrics_mask == masks_pred_cls) & valid_mask
                    pixel_acc += correct.sum().float() / valid_mask.sum().float()
                else:
                    # For binary segmentation
                    true_masks_float = (metrics_mask == 1).float()
                    intersection = (masks_pred * true_masks_float).sum()
                    dice_score += (2.0 * intersection) / (masks_pred.sum() + true_masks_float.sum() + 1e-10)
                    union = (masks_pred + true_masks_float).sum() - intersection
                    iou_score += intersection / (union + 1e-10)
                    
                    correct = ((masks_pred > 0.5) == (true_masks_float > 0.5)).float()
                    pixel_acc += correct.mean()
    
    if n_classes > 1:
        dice_score = dice_per_class.mean().item() / num_batches
        iou_score = iou_per_class.mean().item() / num_batches
        pixel_acc = pixel_acc.item() / num_batches
        dice_per_class = dice_per_class / num_batches
        iou_per_class = iou_per_class / num_batches
    else:
        dice_score = dice_score.item() / num_batches
        iou_score = iou_score.item() / num_batches
        pixel_acc = pixel_acc.item() / num_batches
        dice_per_class = torch.tensor([dice_score], device=device)
        iou_per_class = torch.tensor([iou_score], device=device)
    
    return dice_score, iou_score, pixel_acc, dice_per_class, iou_per_class 