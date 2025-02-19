import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def multiclass_iou(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IoU for all classes
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    union = (input + target).sum(dim=sum_dim) - inter

    iou = (inter + epsilon) / (union + epsilon)
    return iou.mean()


def pixel_accuracy(input: Tensor, target: Tensor):
    # Fraction of correctly predicted pixels
    assert input.size() == target.size()

    correct = (input == target).sum()
    total = input.numel()
    return correct / total

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou = 0
    acc = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                true_masks_processed = mask_true.clone()
                true_masks_processed[true_masks_processed == 255] = 0
                assert true_masks_processed.min() >= 0 and true_masks_processed.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                true_masks_processed = F.one_hot(true_masks_processed, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], true_masks_processed[:, 1:], reduce_batch_first=False)

                # compute the IoU
                iou += multiclass_iou(mask_pred[:, 1:], true_masks_processed[:, 1:], reduce_batch_first=False)

                # compute the pixel accuracy
                acc += pixel_accuracy(mask_pred.argmax(dim=1), true_masks_processed.argmax(dim=1))

    net.train()
    return dice_score / max(num_val_batches, 1), iou / max(num_val_batches, 1), acc / max(num_val_batches, 1)