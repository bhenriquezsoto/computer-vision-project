import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import numpy as np


def compute_dice_per_class(pred: Tensor, target: Tensor, n_classes: int = 3, epsilon: float = 1e-6):
    """Compute per-class Dice score."""
    dice_scores = torch.zeros(n_classes, device=pred.device)

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        dice = (2 * intersection + epsilon) / (pred_inds.sum().float() + target_inds.sum().float() + epsilon)

        dice_scores[cls] = dice

    return dice_scores 


def compute_iou_per_class(pred: Tensor, target: Tensor, n_classes: int = 3, epsilon: float = 1e-6):
    """Compute per-class IoU."""
    iou_scores = torch.zeros(n_classes, device=pred.device)

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou = (intersection + epsilon) / (union + epsilon)

        iou_scores[cls] = iou

    return iou_scores

def compute_pixel_accuracy(pred: Tensor, target: Tensor):
    """Compute pixel accuracy (fraction of correctly classified pixels)."""
    return (pred == target).sum().float() / target.numel()

def dice_loss(input: Tensor, target: Tensor, n_classes: int = 1, epsilon: float = 1e-6):
    """
    Compute Dice loss for binary or multi-class segmentation.

    - Uses sigmoid for binary segmentation.
    - Uses softmax for multi-class segmentation.
    - Computes per-class Dice scores and returns:
        - Mean Dice score if `n_classes > 1` (multi-class).
        - Single Dice score if `n_classes == 1` (binary segmentation).

    Args:
        input (Tensor): Model predictions (logits), shape (B, C, H, W)
        target (Tensor): Ground truth segmentation masks, shape (B, H, W)
        n_classes (int): Number of segmentation classes (default 1 for binary).
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: Dice loss (scalar).
    """
    if n_classes == 1:  # Binary segmentation case
        input = torch.sigmoid(input)  # Convert logits to probabilities
        target = target.float()  # Ensure target is float
        pred = input  # Use raw probability instead of thresholding
    else:  # Multi-class segmentation case
        input = torch.softmax(input, dim=1)  # Convert logits to class probabilities
        target = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()  # Convert to one-hot
        pred = input  # Use probability scores instead of argmax

    # Compute per-class Dice scores
    dice_scores = compute_dice_per_class(pred, target, n_classes=n_classes, epsilon=epsilon)

    # Take mean for multi-class, take scalar for binary
    return 1 - (dice_scores.mean() if n_classes > 1 else dice_scores[0])


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, dim = 256, n_classes=3, desc='Validation round'):
    """
    Evaluates model on the validation dataset.

    Returns:
    - Mean Dice Score
    - Mean IoU
    - Mean Pixel Accuracy
    - Per-class Dice Scores
    - Per-class IoU Scores
    """
    net.eval()
    num_batches = len(dataloader)

    total_dice = torch.zeros(n_classes, device=device)
    total_iou = torch.zeros(n_classes, device=device)
    total_acc = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc=desc, unit='batch', leave=False):        
            image, mask_true = batch['image'], batch['mask']
            
            # Get image size and resize it to the model's input size
            print("image shape", image.shape)
            original_size = image.shape[-2:]
            print("original size", original_size)
            image = F.interpolate(image, size=(dim,dim), mode='bilinear')
            
            print("image shape after interpolation", image.shape)
        
            # Move to correct device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict masks
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)  # Convert to class indices
            
            # Resize masks to original size
            mask_pred = F.interpolate(mask_pred.unsqueeze(1).float(), size=original_size, mode='nearest').long().squeeze(1)

            # Ignore `255` class (void label) in mask
            mask_true = mask_true.clone()
            mask_true[mask_true == 255] = 0  # Treat void label as background

            # Compute Dice Score, IoU, and Pixel Accuracy
            dice_scores = compute_dice_per_class(mask_pred, mask_true, n_classes=n_classes)
            iou_scores = compute_iou_per_class(mask_pred, mask_true, n_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(mask_pred, mask_true)

            # Accumulate results
            total_dice += dice_scores
            total_iou += iou_scores
            total_acc += pixel_acc

    net.train()

    # Compute mean metrics
    mean_dice = total_dice.mean().item() / num_batches
    mean_iou = total_iou.mean().item() / num_batches
    mean_acc = total_acc / num_batches

    return mean_dice, mean_iou, mean_acc, total_dice / num_batches, total_iou / num_batches