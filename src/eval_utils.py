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
def evaluate_segmentation(net, dataloader, device, amp, n_classes=3, class_weights=None, 
                         mode='val', is_point_model=False, desc=None):
    """
    Unified function to evaluate segmentation models for train, validation, and test.
    
    Args:
        net (torch.nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): The dataloader
        device (torch.device): The device to run evaluation on
        amp (bool): Whether to use automatic mixed precision
        n_classes (int): Number of classes for segmentation
        class_weights (torch.Tensor, optional): Class weights to apply
        mode (str): One of 'train', 'val', or 'test' 
        is_point_model (bool): Whether the model is point-based
        desc (str, optional): Description for tqdm progress bar
        
    Returns:
        tuple: A tuple containing:
            - mean_dice: Mean Dice score
            - mean_iou: Mean IoU score 
            - mean_acc: Mean pixel accuracy
            - dice_per_class: Per-class Dice scores
            - iou_per_class: Per-class IoU scores
    """
    # Check if we're evaluating an autoencoder in reconstruction phase
    is_autoencoder_reconstruction = hasattr(net, 'training_phase') and net.training_phase == 'reconstruction'
    
    # If we're in reconstruction phase, return dummy metrics and skip evaluation
    if is_autoencoder_reconstruction:
        dummy_metrics = torch.zeros(n_classes, device=device)
        return 0.0, 0.0, 0.0, dummy_metrics, dummy_metrics
    
    if desc is None:
        desc = f"{mode.capitalize()} evaluation"
    
    net.train(mode == 'train')  # Set model to train mode if mode is 'train', otherwise eval mode
    
    num_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    pixel_acc = 0
    dice_per_class = torch.zeros(n_classes, device=device)
    iou_per_class = torch.zeros(n_classes, device=device)
    
    # Context manager for inference mode (unless in training mode)
    # When evaluating during training, we need to use the current model state without activating inference mode
    cm = torch.no_grad()
    
    with cm:
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            # In training mode with a single batch, skip tqdm for efficiency
            batch_iterator = dataloader if mode == 'train' and num_batches <= 1 else tqdm(dataloader, total=num_batches, desc=desc, unit="batch", leave=False)
            
            for batch in batch_iterator:
                images = batch['image']
                true_masks = batch['mask']
                
                # Move to device
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                # Forward pass
                if is_point_model:
                    point_heatmap = batch['point'].to(device=device, dtype=torch.float32)
                    masks_pred = net(images, point_heatmap)
                else:
                    masks_pred = net(images)
                
                # Create a mask to identify void pixels (255)
                void_pixels = true_masks == 255
                
                # For metrics calculation, create a mask that excludes void pixels
                metrics_mask = true_masks.clone()
                
                # Get predictions - but keep logits for weighted evaluation
                if n_classes == 1:
                    # For binary segmentation
                    masks_pred_probs = torch.sigmoid(masks_pred)
                    masks_pred_binary = (masks_pred_probs > 0.5).float()
                    
                    true_masks_float = (metrics_mask == 1).float()
                    intersection = (masks_pred_binary * true_masks_float).sum()
                    dice_score += (2.0 * intersection) / (masks_pred_binary.sum() + true_masks_float.sum() + 1e-10)
                    union = (masks_pred_binary + true_masks_float).sum() - intersection
                    iou_score += intersection / (union + 1e-10)
                    
                    correct = ((masks_pred_binary > 0.5) == (true_masks_float > 0.5)).float()
                    pixel_acc += correct.mean()
                else:
                    # For multi-class segmentation
                    # Get the hard prediction for the mask
                    masks_pred_cls = masks_pred.argmax(dim=1)
                    
                    # Set void pixels in the metrics mask to match the prediction to exclude them
                    metrics_mask[void_pixels] = masks_pred_cls[void_pixels]
                    
                    # Temporarily set void pixels to a valid class index to avoid errors in one_hot
                    true_masks_one_hot = F.one_hot(metrics_mask, n_classes).permute(0, 3, 1, 2).float()
                    
                    # Get probabilities using softmax
                    masks_pred_probs = F.softmax(masks_pred, dim=1)
                    
                    # Create a mask for valid pixels (not void)
                    valid_mask = ~void_pixels
                    
                    # Compute metrics only on valid pixels
                    for c in range(n_classes):
                        pred_c = masks_pred_probs[:, c]
                        true_c = true_masks_one_hot[:, c]
                        
                        # Apply the valid mask
                        pred_c_valid = pred_c * valid_mask.float()
                        true_c_valid = true_c * valid_mask.float()
                        
                        # Use soft Dice calculation (using probabilities instead of binary)
                        # This better matches how dice_loss works during training
                        intersection = (pred_c_valid * true_c_valid).sum()
                        pred_sum = pred_c_valid.sum()
                        true_sum = true_c_valid.sum()
                        
                        # Calculate Dice score and IoU for this class
                        if (pred_sum + true_sum) > 0:
                            class_dice = (2.0 * intersection) / (pred_sum + true_sum)
                            # Apply class weights if provided
                            if class_weights is not None:
                                class_dice = class_dice * class_weights[c]
                            dice_per_class[c] += class_dice
                        
                        # For IoU, still use the binary prediction for standard calculation
                        pred_binary = (masks_pred_cls == c).float()
                        true_binary = (metrics_mask == c).float()
                        
                        # Calculate binary IoU with valid mask
                        pred_binary_valid = pred_binary * valid_mask.float()
                        true_binary_valid = true_binary * valid_mask.float()
                        
                        intersection_binary = (pred_binary_valid * true_binary_valid).sum()
                        union_binary = pred_binary_valid.sum() + true_binary_valid.sum() - intersection_binary
                        
                        if union_binary > 0:
                            class_iou = intersection_binary / union_binary
                            # Apply class weights if provided
                            if class_weights is not None:
                                class_iou = class_iou * class_weights[c]
                            iou_per_class[c] += class_iou
                    
                    # Calculate overall pixel accuracy excluding void pixels
                    correct = (metrics_mask == masks_pred_cls) & valid_mask
                    pixel_acc += correct.sum().float() / valid_mask.sum().float()
    
    # Compute mean metrics
    if n_classes > 1:
        if class_weights is not None:
            # If using weighted metrics, use sum rather than mean
            mean_dice = dice_per_class.sum().item() / num_batches
            mean_iou = iou_per_class.sum().item() / num_batches
        else:
            mean_dice = dice_per_class.mean().item() / num_batches
            mean_iou = iou_per_class.mean().item() / num_batches
        mean_acc = pixel_acc.item() / num_batches
        dice_per_class = dice_per_class / num_batches
        iou_per_class = iou_per_class / num_batches
    else:
        mean_dice = dice_score.item() / num_batches
        mean_iou = iou_score.item() / num_batches
        mean_acc = pixel_acc.item() / num_batches
        dice_per_class = torch.tensor([mean_dice], device=device)
        iou_per_class = torch.tensor([mean_iou], device=device)
    
    return mean_dice, mean_iou, mean_acc, dice_per_class, iou_per_class 