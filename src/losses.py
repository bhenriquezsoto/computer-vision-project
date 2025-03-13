import torch
import torch.nn.functional as F

def adaptive_focal_loss(inputs, targets, num_masks, epsilon: float = 0.5, gamma: float = 2,
                       delta: float = 0.4, alpha: float = 1.0, eps: float = 1e-12):
    """
    Adaptive Focal Loss for interactive segmentation.
    
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: Number of masks in the batch
        epsilon: Weighting factor in range (0,1) to balance positive vs negative examples
        gamma: Base exponent of the modulating factor
        delta: Factor to estimate the gap between ∇B and BCE loss gradient
        alpha: Coefficient of poly loss
        eps: Term added to the denominator to improve numerical stability
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)

    # Calculate adaptive gamma based on positive samples
    one_hot = targets > 0.5
    with torch.no_grad():
        p_sum = torch.sum(torch.where(one_hot, p_t, 0), dim=-1, keepdim=True)
        ps_sum = torch.sum(torch.where(one_hot, 1, 0), dim=-1, keepdim=True)
        gamma = gamma + (1 - (p_sum / (ps_sum + eps)))

    # Calculate adaptive beta
    beta = (1 - p_t) ** gamma

    # Calculate adaptive multiplier
    with torch.no_grad():
        sw_sum = torch.sum(torch.ones(p_t.shape, device=p_t.device), dim=-1, keepdim=True)
        beta_sum = (1 + delta * gamma) * torch.sum(beta, dim=-1, keepdim=True) + eps
        mult = sw_sum / beta_sum

    # Compute final loss
    loss = mult * ce_loss * beta + alpha * (1 - p_t) ** (gamma + 1)

    # Apply epsilon weighting if specified
    if epsilon >= 0:
        epsilon_t = epsilon * targets + (1 - epsilon) * (1 - targets)
        loss = epsilon_t * loss

    return loss.mean(1).sum() / num_masks

def adaptive_focal_loss_multi_class(inputs, targets, num_masks, class_weights=None, epsilon: float = 0.5, gamma: float = 2,
                                  delta: float = 0.4, alpha: float = 1.0, eps: float = 1e-12):
    """
    Multi-class version of Adaptive Focal Loss.
    
    Args:
        inputs: A float tensor of shape [B, C, H, W] where C is the number of classes
        targets: A long tensor of shape [B, H, W] with class indices
        num_masks: Number of masks in the batch
        class_weights: Optional tensor of class weights [C]
        epsilon: Weighting factor for positive/negative balance
        gamma: Base exponent of the modulating factor
        delta: Factor to estimate the gap between ∇B and BCE loss gradient
        alpha: Coefficient of poly loss
        eps: Term added to the denominator to improve numerical stability
    Returns:
        Loss tensor
    """
    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
    
    # Apply softmax to get probabilities
    probs = F.softmax(inputs, dim=1)
    
    # Calculate cross entropy loss
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    
    # Calculate p_t for each class
    p_t = torch.where(targets_one_hot == 1, probs, 1 - probs)
    
    # Calculate adaptive gamma based on positive samples
    with torch.no_grad():
        p_sum = torch.sum(torch.where(targets_one_hot == 1, p_t, 0), dim=(2, 3), keepdim=True)
        ps_sum = torch.sum(targets_one_hot, dim=(2, 3), keepdim=True)
        gamma = gamma + (1 - (p_sum / (ps_sum + eps)))
    
    # Calculate adaptive beta
    beta = (1 - p_t) ** gamma
    
    # Calculate adaptive multiplier
    with torch.no_grad():
        sw_sum = torch.sum(torch.ones(p_t.shape, device=p_t.device), dim=(2, 3), keepdim=True)
        beta_sum = (1 + delta * gamma) * torch.sum(beta, dim=(2, 3), keepdim=True) + eps
        mult = sw_sum / beta_sum
    
    # Compute final loss with proper dimension handling
    # Average beta over spatial dimensions
    beta_mean = beta.mean(dim=(2, 3))
    
    # Compute polynomial term and average over spatial dimensions
    poly_term = ((1 - p_t) ** (gamma + 1)).mean(dim=(2, 3))  # [B, C]
    
    # Ensure all terms have the same dimensions before combining
    # mult: [B, 1, 1, 1] -> [B, 1]
    # ce_loss: [B, H, W]
    # beta_mean: [B, C]
    # poly_term: [B, C]
    
    # First, average ce_loss over spatial dimensions
    ce_loss = ce_loss.mean(dim=(1, 2))  # [B]
    
    # Expand ce_loss to match beta_mean dimensions
    ce_loss = ce_loss.unsqueeze(1)  # [B, 1]
    
    # Now combine terms with matching dimensions
    loss = mult.squeeze(-1).squeeze(-1) * ce_loss * beta_mean + alpha * poly_term  # [B, C]
    
    # First average over batch dimension
    loss = loss.mean(0)  # [C]
    
    # Then apply class weights if provided
    if class_weights is not None:
        loss = loss * class_weights  # [C]
    
    # Finally, sum over classes and normalize by number of masks
    return loss.sum() / num_masks

    # Apply epsilon weighting if specified
    if epsilon >= 0:
        epsilon_t = epsilon * targets_one_hot + (1 - epsilon) * (1 - targets_one_hot)
        epsilon_t = epsilon_t.mean(dim=(2, 3))  # Average over spatial dimensions
        loss = (epsilon_t * loss).mean(dim=1)  # Average over classes
    
    return loss.sum() / num_masks 