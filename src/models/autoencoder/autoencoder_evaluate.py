import torch
from tqdm import tqdm
from torch import nn
import numpy as np
from skimage.metrics import structural_similarity as ssim


@torch.inference_mode()
def evaluate_autoencoder(net, dataloader, device, amp, desc='Autoencoder Validation'):
    """
    Evaluates autoencoder model on the validation dataset.

    Returns:
    - Mean MSE Loss
    - Mean SSIM (IoU-like metric for reconstruction)
    """
    net.eval()
    num_batches = len(dataloader)

    total_mse = 0
    total_ssim = 0
    
    criterion = nn.MSELoss()

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc=desc, unit='batch', leave=False):        
            image = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # Predict reconstruction
            reconstructed = net(image)

            # Compute MSE loss
            mse_loss = criterion(reconstructed, image)
            total_mse += mse_loss.item()

            # Compute SSIM
            img_np = image.cpu().numpy()
            rec_np = reconstructed.cpu().numpy()
            batch_ssim = np.mean([ssim(img, rec, multichannel=True) for img, rec in zip(img_np, rec_np)])
            total_ssim += batch_ssim

    net.train()

    # Compute mean metrics
    mean_mse = total_mse / num_batches
    mean_ssim = total_ssim / num_batches

    return mean_mse, mean_ssim