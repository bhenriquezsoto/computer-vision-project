import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor


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

def dice_loss(input: Tensor, target: Tensor, n_classes: int = 1, epsilon: float = 1e-6, ignore_index: int = None, class_weights: Tensor = None):
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
        ignore_index (int): Optional label value to ignore (e.g., 255 for void)
        class_weights (Tensor): Optional tensor of class weights [C]

    Returns:
        Tensor: Dice loss (scalar).
    """
    if n_classes == 1:  # Binary segmentation case
        input = torch.sigmoid(input)  # Convert logits to probabilities
        target = target.float()  # Ensure target is float
        pred = input  # Use raw probability instead of thresholding
    else:  # Multi-class segmentation case
        input = torch.softmax(input, dim=1)  # Convert logits to class probabilities
        
        # Create a mask for valid pixels (not ignore_index)
        valid_mask = None
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            # Create a temporary target that won't cause issues with one_hot
            temp_target = target.clone()
            temp_target[~valid_mask] = 0  # Temporarily set ignored pixels to 0
            # Convert to one-hot
            target_one_hot = F.one_hot(temp_target, num_classes=n_classes).permute(0, 3, 1, 2).float()
            # Zero out the pixels that should be ignored in the one-hot encoding
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(1).expand_as(target_one_hot)
                target_one_hot = target_one_hot * valid_mask.float()
        else:
            target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()
        
        target = target_one_hot
        pred = input  # Use probability scores instead of argmax

    # Compute per-class Dice scores with ignore handling
    batch_size = input.size(0)
    
    if n_classes == 1:
        # For binary case, flatten and apply dice calculation
        input_flat = input.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        # Apply mask if ignore_index is provided
        if ignore_index is not None and valid_mask is not None:
            valid_mask_flat = valid_mask.view(batch_size, -1)
            intersection = torch.sum(input_flat * target_flat * valid_mask_flat.float(), dim=1)
            total = torch.sum(input_flat * valid_mask_flat.float(), dim=1) + torch.sum(target_flat * valid_mask_flat.float(), dim=1)
        else:
            intersection = torch.sum(input_flat * target_flat, dim=1)
            total = torch.sum(input_flat, dim=1) + torch.sum(target_flat, dim=1)
            
        dice = (2.0 * intersection + epsilon) / (total + epsilon)
        return 1 - dice.mean()
    else:
        # For multi-class, compute dice per channel/class
        dice_per_class = []
        
        for class_idx in range(n_classes):
            # Extract class predictions and targets
            pred_class = pred[:, class_idx, :, :]
            target_class = target[:, class_idx, :, :]
            
            # Flatten for easier computation
            pred_class_flat = pred_class.view(batch_size, -1)
            target_class_flat = target_class.view(batch_size, -1)
            
            # Handle ignore_index if provided
            if ignore_index is not None and valid_mask is not None:
                valid_mask_flat = valid_mask[:, class_idx, :, :].view(batch_size, -1)
                intersection = torch.sum(pred_class_flat * target_class_flat * valid_mask_flat.float(), dim=1)
                total = torch.sum(pred_class_flat * valid_mask_flat.float(), dim=1) + torch.sum(target_class_flat * valid_mask_flat.float(), dim=1)
            else:
                intersection = torch.sum(pred_class_flat * target_class_flat, dim=1)
                total = torch.sum(pred_class_flat, dim=1) + torch.sum(target_class_flat, dim=1)
            
            # Compute Dice and handle cases where both prediction and target are empty
            class_dice = (2.0 * intersection + epsilon) / (total + epsilon)
            dice_per_class.append(class_dice.mean())
        
        # Stack dice scores and apply class weights if provided
        dice_scores = torch.stack(dice_per_class)
        if class_weights is not None:
            dice_scores = dice_scores * class_weights
        
        # Return weighted mean Dice loss across all classes
        return 1 - torch.mean(dice_scores)


@torch.inference_mode()
def compute_metrics(net, dataloader, device, amp, dim = 256, n_classes=3, desc='Validation round'):
    """
    Computes metrics for a model on a dataset.

    Returns:
    - Mean Dice Score
    - Mean IoU
    - Mean Pixel Accuracy
    - Per-class Dice Scores
    - Per-class IoU Scores
    """
    net.eval()
    num_batches = len(dataloader)
    
    # Check if we're evaluating an autoencoder in reconstruction phase
    is_autoencoder_reconstruction = hasattr(net, 'training_phase') and net.training_phase == 'reconstruction'
    
    # If we're in reconstruction phase, return dummy metrics and skip evaluation
    if is_autoencoder_reconstruction:
        dummy_metrics = torch.zeros(n_classes, device=device)
        return 0.0, 0.0, 0.0, dummy_metrics, dummy_metrics

    total_dice = torch.zeros(n_classes, device=device)
    total_iou = torch.zeros(n_classes, device=device)
    total_acc = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc=desc, unit='batch', leave=False):        
            image, mask_true = batch['image'], batch['mask']
            
            # Get original size from the mask
            original_size = mask_true.shape[-2:]
        
            # Move to correct device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict masks
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)  # Convert to class indices
            
            # Resize masks to original size
            mask_pred = F.interpolate(mask_pred.unsqueeze(1).float(), size=original_size, mode='nearest').long().squeeze(1)

            # Create a mask to identify void pixels (255)
            void_pixels = mask_true == 255
            
            # For metrics calculation, create a mask that excludes void pixels
            metrics_mask = mask_true.clone()
            
            # Set void pixels in the metrics mask to match the prediction
            # This effectively ignores these pixels in the metrics calculation
            metrics_mask[void_pixels] = mask_pred[void_pixels]

            # Compute Dice Score, IoU, and Pixel Accuracy
            dice_scores = compute_dice_per_class(mask_pred, metrics_mask, n_classes=n_classes)
            iou_scores = compute_iou_per_class(mask_pred, metrics_mask, n_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(mask_pred, metrics_mask)

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