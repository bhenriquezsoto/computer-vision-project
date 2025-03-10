
import logging
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.metrics import structural_similarity as ssim

import wandb
from models.autoencoder.autoencoder_evaluate import evaluate_autoencoder


# import data loading
from data_loading import AutoencoderDataset


def train_autoencoder_model(
        model,
        device,
        optimizer,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = False,
        img_dim: int = 256,
        amp: bool = False,
        gradient_clipping: float = 1.0
):
    
    experiment = wandb.init(project='CV', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_dim=img_dim, model="Autoencoder", amp=amp, optimizer=optimizer))
    
    
    """Trains the autoencoder model for image reconstruction."""
    logging.info("""Starting autoencoder training:
            Model:           {model.__class__.__name__} 
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Weight decay:    {weight_decay}
            Optimizer:       {optimizer}
            Training size:   {len(train_set)}
            Validation size: {len(val_set)}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Image dimensions:{img_dim}x{img_dim}
            Mixed Precision: {amp}
    """)
    
    # Load dataset (images only)
    all_images = list(dir_img.glob('*'))
    train_images, val_images = train_test_split(all_images, test_size=val_percent, random_state=42)
    train_set = AutoencoderDataset(train_images, dim=img_dim)
    val_set = AutoencoderDataset(val_images, dim=img_dim)
    
    # Create data loaders
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, **loader_args)
    
    # Set up optimizer, loss, scheduler, and AMP
    optimizer = optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) # or ReduceLROnPlateau(optimizer, mode='min', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    
    global_step = 0
    best_val_ssim = -1
    best_val_ssim_after_epoch_10 = -1
    
    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        total_mse = 0
        total_ssim = 0
        
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img', leave=True) as pbar:
            for batch in train_loader:
                images = batch['image']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    reconstructed = model(images)
                    mse_loss = criterion(reconstructed, images)
                    total_mse += mse_loss.item()
                    
                    img_np = images.cpu().numpy()
                    rec_np = reconstructed.cpu().numpy()
                    batch_ssim = np.mean([ssim(img, rec, multichannel=True) for img, rec in zip(img_np, rec_np)])
                    total_ssim += batch_ssim
                    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(mse_loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                pbar.update(images.shape[0])
                global_step += 1
                
                # Log training metrics
                experiment.log({
                    'train MSE Loss': mse_loss.item(),
                    'train SSIM': batch_ssim,
                    'step': global_step,
                    'epoch': epoch
                })
                
                pbar.set_postfix(mse_loss=f"{mse_loss.item():.4f}", ssim=f"{batch_ssim:.4f}")
    
        # Compute average training metrics
        epoch_loss = total_mse / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        
        
        logging.info(f"Epoch {epoch} - Training MSE Loss: {epoch_loss:.4f}, SSIM: {avg_ssim:.4f}")
        
        # Perform validation at the end of each epoch
        val_mse, val_ssim = evaluate_autoencoder(model, val_loader, device, amp=amp)
        
        # Update scheduler (if using ReduceLROnPlateau)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mse)
        else:
            scheduler.step()
            
        # Log validation metrics
        experiment.log({
            'validation MSE Loss': val_mse,
            'validation SSIM': val_ssim,
            'epoch': epoch
        })
        
        logging.info(f"Epoch {epoch} - Validation MSE Loss: {val_mse:.4f}, SSIM: {val_ssim:.4f}")
        
        
        # Save the best model based on validation SSIM
        if val_ssim > best_val_ssim and epoch <= 10:
            best_val_ssim = val_ssim
            run_name = wandb.run.name
            model_path = os.path.join(dir_checkpoint, f'best_autoencoder_{run_name}.pth')
            state_dict = {"model_state_dict": model.state_dict()}
            torch.save(state_dict, model_path)
            logging.info(f'Best autoencoder model saved as {model_path}!')
        
        # Save the best model based on validation SSIM after epoch 10 to avoid only saving early peaks
        if epoch > 10 and val_ssim > best_val_ssim_after_epoch_10:
            best_val_ssim_after_epoch_10 = val_ssim
            run_name = wandb.run.name
            model_path = os.path.join(dir_checkpoint, f'best_autoencoder_after_epoch_10_{run_name}.pth')
            state_dict = {"model_state_dict": model.state_dict()}
            torch.save(state_dict, model_path)
            logging.info(f'Best autoencoder model after epoch 10 saved as {model_path}!')
            
        # Optionally save checkpoint every epoch
        if save_checkpoint or epoch == epochs or epoch % 50 == 0:
            checkpoint_path = os.path.join(dir_checkpoint, f'checkpoint_autoencoder_epoch{epoch}.pth')
            state_dict = {"model_state_dict": model.state_dict()}
            torch.save(state_dict, checkpoint_path)
            logging.info(f'Autoencoder checkpoint saved at {checkpoint_path}')
            
    # After all epochs are completed, evaluate on the test set
    logging.info("Training complete. Evaluating on test set...")
    
    # Load the best saved model
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    logging.info(f'Model loaded from {model_path}')
    model.to(device)
    model.eval()
    
    # Load test dataset
    test_img_files = list(dir_test_img.glob('*'))
    test_dataset = AutoencoderDataset(test_img_files, dim=img_dim)  # Use same preprocessing as training
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_args)
    
    # Evaluate on the test set
    test_mse, test_ssim = evaluate_autoencoder(model, test_loader, device, amp=amp, desc='Testing round')
    
    # Print test results
    logging.info(f"Test MSE Loss: {test_mse:.4f}")
    logging.info(f"Test SSIM: {test_ssim:.4f}")
    
    logging.info("Test evaluation complete.")
    
    # Optional: Log test results to wandb
    experiment.log({
        "test MSE Loss": test_mse,
        "test SSIM": test_ssim
    })