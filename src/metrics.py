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

        # Check if this class exists in the target (to avoid NaN for classes not present)
        if target_inds.sum() == 0:
            # If class doesn't exist in target, we can:
            # 1. Give it a score of 1.0 if pred also doesn't have it
            # 2. Give it a score of 0.0 if pred incorrectly predicts it
            # 3. Skip it completely (we'll use option 1 here)
            dice_scores[cls] = 1.0 if pred_inds.sum() == 0 else 0.0
            continue

        intersection = (pred_inds & target_inds).sum().float()
        pred_sum = pred_inds.sum().float()
        target_sum = target_inds.sum().float()
        
        # Modified Dice calculation that's more stable for small objects
        dice = (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
        
        dice_scores[cls] = dice

    return dice_scores 


def compute_iou_per_class(pred: Tensor, target: Tensor, n_classes: int = 3, epsilon: float = 1e-6):
    """Compute per-class IoU."""
    iou_scores = torch.zeros(n_classes, device=pred.device)

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        # Check if this class exists in the target (to avoid NaN for classes not present)
        if target_inds.sum() == 0:
            # Similar handling as in compute_dice_per_class
            iou_scores[cls] = 1.0 if pred_inds.sum() == 0 else 0.0
            continue

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
def compute_metrics(net, dataloader, device, amp, dim = 256, n_classes=3, desc='Validation round', is_point_model=False):
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
    
    # Track number of batches each class appears in
    class_batch_count = torch.zeros(n_classes, device=device, dtype=torch.float32)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc=desc, unit='batch', leave=False):        
            image, mask_true = batch['image'], batch['mask']
            
            # Get original size from the mask
            original_size = mask_true.shape[-2:]
        
            # Move to correct device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict masks based on model type
            if is_point_model:
                point_heatmap = batch['point'].to(device=device, dtype=torch.float32)
                mask_pred = net(image, point_heatmap)
            else:
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

            # Check which classes are present in this batch
            class_present = torch.zeros(n_classes, device=device, dtype=torch.bool)
            for cls in range(n_classes):
                class_present[cls] = (mask_true == cls).any()
            
            # Only accumulate metrics for classes that are present
            total_dice += torch.where(class_present, dice_scores, torch.zeros_like(dice_scores))
            total_iou += torch.where(class_present, iou_scores, torch.zeros_like(iou_scores))
            class_batch_count += class_present.float()
            
            total_acc += pixel_acc

    net.train()

    # Avoid division by zero for classes that never appeared
    class_batch_count = torch.maximum(class_batch_count, torch.ones_like(class_batch_count))
    
    # Compute per-class metrics
    per_class_dice = total_dice / class_batch_count
    per_class_iou = total_iou / class_batch_count
    
    # Compute mean metrics
    mean_dice = per_class_dice.mean().item()
    mean_iou = per_class_iou.mean().item()
    mean_acc = total_acc / num_batches

    return mean_dice, mean_iou, mean_acc, per_class_dice, per_class_iou


def sigmoid_adaptive_focal_loss(inputs, targets, num_masks, epsilon: float = 0.5, gamma: float = 2,
                                delta: float = 0.4, alpha: float = 1.0, eps: float = 1e-12):
    """
    Adaptive Focal Loss from AdaptiveClick paper.
    
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        epsilon: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        delta: A Factor in range (0,1) to estimate the gap between the term of ∇B
                and the gradient term of bce loss.
        alpha: A coefficient of poly loss.
        eps: Term added to the denominator to improve numerical stability.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)

    one_hot = targets > 0.5
    with torch.no_grad():
        p_sum = torch.sum(torch.where(one_hot, p_t, 0), dim=-1, keepdim=True)
        ps_sum = torch.sum(torch.where(one_hot, 1, 0), dim=-1, keepdim=True)
        gamma = gamma + (1 - (p_sum / (ps_sum + eps)))

    beta = (1 - p_t) ** gamma

    with torch.no_grad():
        sw_sum = torch.sum(torch.ones(p_t.shape, device=p_t.device), dim=-1, keepdim=True)
        beta_sum = (1 + delta * gamma) * torch.sum(beta, dim=-1, keepdim=True) + eps
        mult = sw_sum / beta_sum

    loss = mult * ce_loss * beta + alpha * (1 - p_t) ** (gamma + 1)

    if epsilon >= 0:
        epsilon_t = epsilon * targets + (1 - epsilon) * (1 - targets)
        loss = epsilon_t * loss

    return loss.mean(1).sum() / num_masks


def adaptive_focal_loss_multiclass(inputs, targets, num_masks, class_weights=None, epsilon: float = 0.5, gamma: float = 2,
                          delta: float = 0.4, alpha: float = 1.0, eps: float = 1e-12):
    """
    Multi-class version of Adaptive Focal Loss with class weights support.
    
    Args:
        inputs: A float tensor of shape (B, C, H, W) with class logits
        targets: A long tensor of shape (B, H, W) with class indices
        num_masks: Number of masks (usually batch size)
        class_weights: Optional tensor of weights for each class
        epsilon, gamma, delta, alpha, eps: Same as in sigmoid_adaptive_focal_loss
        
    Returns:
        Loss tensor
    """
    # Get number of classes from inputs
    n_classes = inputs.shape[1]
    
    # Debug: Check for invalid values in targets
    unique_values = torch.unique(targets)
    if torch.any(unique_values >= n_classes) and torch.any(unique_values != 255):
        print(f"WARNING: Found invalid values in targets: {unique_values}")
    
    # Handle void label (255) in targets before one-hot encoding
    # Create a copy of targets to avoid modifying the original
    targets_processed = targets.clone()
    # Set void label to 0 (background) to avoid index out of bounds
    targets_processed[targets == 255] = 0
    
    # One-hot encode targets (now all values are valid indices)
    targets_one_hot = F.one_hot(targets_processed, num_classes=n_classes).permute(0, 3, 1, 2).float()
    
    # Create a mask for valid (non-void) pixels
    valid_mask = (targets != 255).float().unsqueeze(1).expand_as(targets_one_hot)
    
    # Zero out the one-hot encoding for void pixels
    targets_one_hot = targets_one_hot * valid_mask
    
    # Apply softmax to get class probabilities
    inputs_soft = F.softmax(inputs, dim=1)
    
    # Initialize loss
    total_loss = 0
    
    # Calculate loss for each class
    for cls in range(n_classes):
        cls_inputs = inputs[:, cls]  # (B, H, W)
        cls_targets = targets_one_hot[:, cls]  # (B, H, W)
        
        # Calculate class-specific loss
        cls_loss = sigmoid_adaptive_focal_loss(
            cls_inputs, cls_targets, num_masks, 
            epsilon=epsilon, gamma=gamma, delta=delta, alpha=alpha, eps=eps
        )
        
        # Apply class weight if provided
        if class_weights is not None and cls < len(class_weights):
            cls_loss = cls_loss * class_weights[cls]
        
        total_loss += cls_loss
    
    # Return total loss (normalization already done in sigmoid_adaptive_focal_loss)
    return total_loss