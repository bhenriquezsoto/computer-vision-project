#!/usr/bin/env python3
"""
Specialized trainer for the AutoencoderSegmentation model.
This handles both reconstruction and segmentation training phases.
"""
import os
import logging
import torch
import torch.nn as nn

# Try to import wandb, but don't fail if it's not available
try:
    import wandb
except ImportError:
    logging.warning("wandb not found. Some logging features will be disabled.")
    wandb = None

from pathlib import Path

from models.base.trainer import BaseTrainer
from models.base.registry import register_model_trainer

@register_model_trainer('AutoencoderSegmentation')
class AutoencoderTrainer(BaseTrainer):
    """Specialized trainer for AutoencoderSegmentation with dual phase training."""
    
    def __init__(
        self,
        model,
        args,
        device,
        dataset_dir=None,
        checkpoint_dir=None,
        project_name='AutoEncoder-Segmentation',
        gradient_accumulation_steps=1,
        early_stopping_patience=10,
        early_stopping_metric='val_iou',
        early_stopping_mode='max',
        val_frequency=1,
        mode=None,
        epochs_recon=None,
        epochs_seg=None,
        freeze_ratio=None,
        use_wandb=True
    ):
        """Initialize the autoencoder trainer with phase-specific parameters."""
        super().__init__(
            model=model,
            args=args,
            device=device,
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            project_name=project_name,
            gradient_accumulation_steps=gradient_accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_mode=early_stopping_mode,
            val_frequency=val_frequency,
            use_wandb=use_wandb
        )
        
        # Get autoencoder specific parameters
        # Mode determines training approach: 'both', 'reconstruction', or 'segmentation'
        try:
            self.mode = mode if mode is not None else getattr(args, 'mode', 'both')
            if self.mode not in ['both', 'reconstruction', 'segmentation']:
                logging.warning(f"Invalid mode '{self.mode}'. Using default 'both'")
                self.mode = 'both'
        except Exception as e:
            logging.error(f"Error setting mode: {str(e)}. Using default 'both'")
            self.mode = 'both'
        
        # Phase-specific parameters
        self.epochs_recon = epochs_recon or getattr(args, 'epochs_recon', 40)
        self.epochs_seg = epochs_seg or getattr(args, 'epochs_seg', 40)
        self.freeze_ratio = freeze_ratio or getattr(args, 'freeze_ratio', 0.8)
        
        # Track phase status
        self.completed_recon = False
        
        # Set initial model mode based on training mode
        if self.mode == 'reconstruction':
            self.model.set_reconstruction_mode()
            self.early_stopping_metric = 'val_loss'
            self.early_stopping_mode = 'min'
        elif self.mode == 'segmentation':
            self.model.set_segmentation_mode(freeze_encoder_ratio=self.freeze_ratio)
            self.early_stopping_metric = 'val_iou'
            self.early_stopping_mode = 'max'
        else:
            # Start with reconstruction for 'both' mode
            self.model.set_reconstruction_mode()
        
        logging.info(f"AutoencoderTrainer initialized with mode: {self.mode}")
        logging.info(f"Training epochs: Reconstruction={self.epochs_recon}, Segmentation={self.epochs_seg}")
    
    def setup_logging(self):
        """Set up experiment logging with autoencoder-specific information."""
        super().setup_logging()
        
        # Add autoencoder-specific config to wandb
        if self.experiment is not None:
            self.experiment.config.update({
                'mode': self.mode,
                'current_model_mode': self.model.mode,
                'bilinear': self.model.bilinear,
                'freeze_ratio': self.freeze_ratio,
                'epochs_recon': self.epochs_recon if self.mode in ['both', 'reconstruction'] else 0,
                'epochs_seg': self.epochs_seg if self.mode in ['both', 'segmentation'] else 0,
            })
        
        # Additional autoencoder-specific logging
        if self.model.mode == 'reconstruction':
            logging.info('''AutoencoderSegmentation - Reconstruction Phase:
                Training to reconstruct input images
                Loss: MSE
                Input channels: {}
                Learning rate: {}
            '''.format(self.model.n_channels, self.learning_rate))
        else:
            logging.info('''AutoencoderSegmentation - Segmentation Phase:
                Training to segment images into {} classes
                Loss: CrossEntropy + Dice
                Encoder frozen: {}%
                Learning rate: {}
            '''.format(self.model.n_classes, self.freeze_ratio * 100, self.learning_rate))
    
    def setup_training(self):
        """Setup training components with appropriate loss function based on mode."""
        super().setup_training()
        
        # Use different loss functions based on the current mode
        if self.mode == 'reconstruction' or (self.mode == 'both' and not hasattr(self, 'completed_recon')):
            # Reconstruction Phase: Use MSE loss
            self.criterion = nn.MSELoss()
            logging.info("Using MSE loss for reconstruction phase")
        else:
            # Segmentation Phase: Use CrossEntropy + Dice loss
            from evaluate import dice_loss
            
            def combined_loss(pred, target):
                bce_loss = nn.CrossEntropyLoss()(pred, target)
                dice = dice_loss(pred, target, n_classes=self.model.n_classes)
                return bce_loss + dice
            
            self.criterion = combined_loss
            logging.info("Using CrossEntropy + Dice loss for segmentation phase")
    
    def model_specific_setup(self):
        """Autoencoder-specific setup steps."""
        # Initialize validation score threshold based on current mode
        # For reconstruction: lower is better (loss)
        # For segmentation: higher is better (IoU)
        try:
            self.best_val_score = float('inf') if self.model.mode == 'reconstruction' else float('-inf')
            self.best_val_score_after_epoch_10 = float('inf') if self.model.mode == 'reconstruction' else float('-inf')
        except Exception as e:
            logging.error(f"Error in model_specific_setup: {str(e)}")
            # Use safe defaults
            self.best_val_score = float('inf')
            self.best_val_score_after_epoch_10 = float('inf')
            
        # Load pretrained encoder weights for segmentation mode
        if self.mode == 'segmentation' and hasattr(self.args, 'load_encoder') and self.args.load_encoder:
            self._load_pretrained_encoder()
            
    def _load_pretrained_encoder(self):
        """Load pretrained encoder weights for segmentation mode."""
        try:
            if not self.args.load_encoder:
                logging.warning("No encoder weights specified for segmentation mode")
                return
                
            logging.info(f"Loading pretrained encoder weights from {self.args.load_encoder}")
            state_dict = torch.load(self.args.load_encoder, map_location=self.device)
            
            # Extract model state dict if it's wrapped
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            # Filter for encoder weights
            encoder_state = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
            
            if not encoder_state:
                logging.warning(f"No encoder weights found in {self.args.load_encoder}")
                return
                
            # Load weights
            missing_keys, unexpected_keys = self.model.encoder.load_state_dict(encoder_state, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing encoder keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected encoder keys: {unexpected_keys}")
                
            logging.info(f"Successfully loaded pretrained encoder weights")
            
        except Exception as e:
            logging.error(f"Failed to load encoder weights: {str(e)}")
            raise
    
    def get_batch_inputs(self, batch):
        """Extract inputs from batch based on model requirements."""
        # Same input for both modes
        return batch['image'].to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
    
    def get_batch_targets(self, batch):
        """Extract targets from batch based on model requirements."""
        if self.model.mode == 'reconstruction':
            # For reconstruction, the target is the input image itself
            return batch['image'].to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
        else:
            # For segmentation, the target is the mask
            return batch['mask'].to(device=self.device, dtype=torch.long)
    
    def compute_loss(self, outputs, targets):
        """Compute loss between model outputs and targets."""
        if self.model.mode == 'reconstruction':
            # For reconstruction, use MSE loss
            return self.criterion(outputs, targets)
        else:
            # For segmentation, use CrossEntropy loss + Dice loss
            return super().compute_loss(outputs, targets)
    
    def compute_metrics(self, outputs, targets):
        """Compute training metrics based on current mode."""
        if self.model.mode == 'reconstruction':
            # For reconstruction, calculate Mean Squared Error, PSNR and SSIM
            with torch.no_grad():
                mse = nn.MSELoss()(outputs, targets).item()
                # PSNR calculation
                max_pixel = 1.0  # Assuming normalized images with range [0,1]
                psnr = 10 * torch.log10(max_pixel**2 / torch.mean((outputs - targets)**2))
                
                # Basic SSIM approximation (simplified)
                # For a complete SSIM implementation, consider using a library like kornia
                def ssim_approx(pred, target, window_size=11):
                    # Get batch size
                    batch_size = pred.size(0)
                    
                    # Constants for stability
                    C1 = (0.01 * max_pixel) ** 2
                    C2 = (0.03 * max_pixel) ** 2
                    
                    # Calculate means
                    mu_x = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)(pred)
                    mu_y = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)(target)
                    
                    # Calculate variance and covariance
                    sigma_x_sq = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)(pred * pred) - mu_x * mu_x
                    sigma_y_sq = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)(target * target) - mu_y * mu_y
                    sigma_xy = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)(pred * target) - mu_x * mu_y
                    
                    # SSIM formula
                    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x_sq + sigma_y_sq + C2))
                    return ssim_map.mean()
                
                try:
                    # Calculate SSIM per batch then average
                    ssim_val = ssim_approx(outputs, targets).item()
                except Exception as e:
                    logging.warning(f"Error calculating SSIM: {str(e)}. Using default value.")
                    ssim_val = 0.0
            
            return {
                'dice_scores': torch.zeros(self.model.n_classes, device=self.device),  # Dummy segmentation metrics
                'iou_scores': torch.zeros(self.model.n_classes, device=self.device),   # Dummy segmentation metrics
                'pixel_acc': 0.0,                                                     # Dummy segmentation metrics
                'mse': mse,                                                           # Mean Squared Error
                'psnr': psnr.item(),                                                  # Peak Signal-to-Noise Ratio
                'ssim': ssim_val                                                      # Structural Similarity Index
            }
        else:
            # For segmentation, use standard metrics
            return super().compute_metrics(outputs, targets)
    
    def log_metrics(self, metrics, loss, batch_idx, epoch, phase='train'):
        """Log metrics based on current mode."""
        if self.model.mode == 'reconstruction':
            # For reconstruction, log reconstruction-specific metrics
            log_dict = {
                f'{phase} reconstruction loss': loss.item(),
                'step': self.global_step,
                'epoch': epoch
            }
            
            # Add reconstruction metrics if available
            if 'mse' in metrics:
                log_dict[f'{phase} MSE'] = metrics['mse']
            if 'psnr' in metrics:
                log_dict[f'{phase} PSNR'] = metrics['psnr']
            if 'ssim' in metrics:
                log_dict[f'{phase} SSIM'] = metrics['ssim']
                
            # Log to wandb
            self.experiment.log(log_dict)
            
            # Store in metrics dict
            for k, v in log_dict.items():
                self.metrics[k] = v
                
            # Display at regular intervals
            if batch_idx % 10 == 0:
                metrics_display = f"MSE: {metrics.get('mse', 0):.4f}, PSNR: {metrics.get('psnr', 0):.2f}dB"
                if 'ssim' in metrics:
                    metrics_display += f", SSIM: {metrics['ssim']:.4f}"
                logging.info(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] - Loss: {loss.item():.4f}, {metrics_display}")
        else:
            # For segmentation, use standard metrics logging
            super().log_metrics(metrics, loss, batch_idx, epoch, phase)
    
    def validate(self, epoch):
        """Validate the model based on current mode."""
        self.model.eval()
        
        if self.model.mode == 'reconstruction':
            # Specialized validation for reconstruction mode
            val_loss = 0
            val_mse = 0
            val_psnr = 0
            val_ssim = 0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    # Get inputs (and use them as targets for reconstruction)
                    inputs = self.get_batch_inputs(batch)
                    targets = self.get_batch_targets(batch)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Loss
                    loss = self.compute_loss(outputs, targets)
                    val_loss += loss.item()
                    
                    # Metrics
                    metrics = self.compute_metrics(outputs, targets)
                    val_mse += metrics.get('mse', 0)
                    val_psnr += metrics.get('psnr', 0)
                    if 'ssim' in metrics:
                        val_ssim += metrics['ssim']
            
            # Average metrics
            val_size = len(self.val_loader)
            val_loss /= val_size
            val_mse /= val_size
            val_psnr /= val_size
            val_ssim /= val_size
            
            # Log validation metrics
            val_metrics = {
                'mse': val_mse,
                'psnr': val_psnr,
                'ssim': val_ssim,
                # Include dummy segmentation metrics for compatibility
                'dice_scores': torch.zeros(self.model.n_classes, device=self.device),
                'iou_scores': torch.zeros(self.model.n_classes, device=self.device),
                'pixel_acc': 0.0
            }
            
            # Log metrics
            self.log_metrics(val_metrics, torch.tensor(val_loss), 0, epoch, phase='validation')
            
            # Display results
            logging.info(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, PSNR: {val_psnr:.2f}dB, SSIM: {val_ssim:.4f}")
            
            # Save best model based on validation loss (lower is better for reconstruction)
            if val_loss < self.best_val_score:
                self.save_best_model(epoch, val_loss, prefix="best_reconstruction")
                self.best_val_score = val_loss
                self.early_stopping_counter = 0
                logging.info(f"Validation loss improved to {val_loss:.4f}. Saving model...")
            else:
                self.early_stopping_counter += 1
                logging.info(f"Validation loss did not improve from {self.best_val_score:.4f}. Early stopping: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
            # After epoch 10, track best model separately
            if epoch > 10 and val_loss < self.best_val_score_after_epoch_10:
                self.save_best_model(epoch, val_loss, prefix="best_reconstruction_after_epoch_10")
                self.best_val_score_after_epoch_10 = val_loss
                
            return val_loss
        else:
            # For segmentation, use standard validation
            return super().validate(epoch)
    
    def train(self):
        """Train the model with support for multi-phase training."""
        if self.mode == 'both':
            # Train both phases sequentially
            logging.info("\n=== Starting Phase A: Reconstruction Training ===")
            self.model.set_reconstruction_mode()
            
            # Set up data and logging
            self.setup()
            
            # Train reconstruction phase
            for epoch in range(1, self.epochs_recon + 1):
                self.train_epoch(epoch)
                if epoch % self.val_frequency == 0:
                    self.validate(epoch)
                self.save_checkpoint(epoch)
            
            # Save reconstruction weights for segmentation phase
            recon_path = os.path.join(self.checkpoint_dir, f'reconstruction_weights_{wandb.run.name}.pth')
            torch.save({"model_state_dict": self.model.state_dict()}, recon_path)
            logging.info(f"Reconstruction weights saved to {recon_path}")
            
            # Mark reconstruction phase as completed
            self.completed_recon = True
            
            # Reset for segmentation phase
            logging.info("\n=== Starting Phase B: Segmentation Training ===")
            self.model.set_segmentation_mode(freeze_encoder_ratio=self.freeze_ratio)
            
            # Reset training state for segmentation
            self.setup_training()
            self.global_step = 0
            self.best_val_score = float('-inf')
            self.best_val_score_after_epoch_10 = float('-inf')
            
            # Train segmentation phase
            for epoch in range(1, self.epochs_seg + 1):
                self.train_epoch(epoch)
                if epoch % self.val_frequency == 0:
                    self.validate(epoch)
                self.save_checkpoint(epoch)
            
            # Evaluate on test set with segmentation model
            self.evaluate_test_set()
            
            return self.model, self.model_paths
            
        elif self.mode == 'reconstruction':
            # Train only reconstruction phase
            logging.info("\n=== Starting Reconstruction Training ===")
            self.model.set_reconstruction_mode()
            
            # Standard single-phase training
            return super().train()
            
        else:  # segmentation mode
            # Train only segmentation phase
            logging.info("\n=== Starting Segmentation Training ===")
            
            # Load pretrained encoder weights if specified
            if hasattr(self.args, 'load_encoder') and self.args.load_encoder:
                self._load_pretrained_encoder()
            
            # Set segmentation mode and freeze encoder layers
            self.model.set_segmentation_mode(freeze_encoder_ratio=self.freeze_ratio)
            
            # Standard single-phase training
            return super().train() 