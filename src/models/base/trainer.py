#!/usr/bin/env python3
"""
Enhanced base trainer class for all segmentation models.

This trainer uses utility modules to handle common tasks like:
- Data loading and preprocessing (data_loading.py)
- Metrics computation and evaluation (evaluate.py)
- Checkpoint management (utils/metrics.py)
- Training utilities (utils/trainer_utils.py)

This design allows the BaseTrainer to be focused on orchestrating the training 
workflow while delegating specific functionality to specialized modules.
"""
import os
import logging
import wandb
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import traceback
import time

# Import our modules for reusing functionality
from data_loading import SegmentationDataset, TestSegmentationDataset
from evaluate import evaluate, compute_dice_per_class, compute_iou_per_class, compute_pixel_accuracy, dice_loss
from utils.metrics import get_best_checkpoint_path, load_model_weights, save_checkpoint, log_test_results
from utils.trainer_utils import setup_data_loaders, setup_optimizer, setup_scheduler, freeze_model_layers, setup_logging, get_training_config

class BaseTrainer:
    """Base trainer for segmentation models with efficient workflow management."""
    
    def __init__(
        self,
        model,
        args,
        device,
        dataset_dir=None,
        checkpoint_dir=None,
        project_name=None,
        gradient_accumulation_steps=1,
        early_stopping_patience=10,
        early_stopping_metric='val_iou',
        early_stopping_mode='max',
        val_frequency=1,
        use_wandb=True
    ):
        """
        Initialize the trainer with model and training parameters.
        
        Args:
            model: The model to train
            args: Command line arguments
            device: Device to run training on
            dataset_dir: Path to dataset directory (containing Train and Test folders)
            checkpoint_dir: Path to save checkpoints to
            project_name: Name of the wandb project
            gradient_accumulation_steps: Number of steps to accumulate gradients
            early_stopping_patience: Number of epochs with no improvement before stopping
            early_stopping_metric: Metric to use for early stopping
            early_stopping_mode: 'max' or 'min' for early stopping
            val_frequency: Validate every N epochs
            use_wandb: Whether to use wandb logging
        """
        # Validate critical parameters
        if model is None:
            raise ValueError("Model cannot be None")
        if args is None:
            raise ValueError("Args cannot be None")
        if device is None:
            raise ValueError("Device cannot be None")
        if early_stopping_mode not in ['max', 'min']:
            logging.warning(f"Invalid early_stopping_mode: {early_stopping_mode}. Using 'max'.")
            early_stopping_mode = 'max'
        if gradient_accumulation_steps < 1:
            logging.warning(f"Invalid gradient_accumulation_steps: {gradient_accumulation_steps}. Using 1.")
            gradient_accumulation_steps = 1
        if early_stopping_patience < 0:
            logging.warning(f"Invalid early_stopping_patience: {early_stopping_patience}. Using 10.")
            early_stopping_patience = 10
        if val_frequency < 1:
            logging.warning(f"Invalid val_frequency: {val_frequency}. Using 1.")
            val_frequency = 1
        
        # Store basic parameters
        self.model = model
        self.args = args
        self.device = device
        
        # Ensure paths are Path objects for OS compatibility
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training hyperparameters from args with comprehensive defaults
        self.batch_size = getattr(args, 'batch_size', 16)
        self.epochs = getattr(args, 'epochs', 50)
        self.learning_rate = getattr(args, 'lr', getattr(args, 'learning_rate', 1e-4))  # Support both lr and learning_rate
        self.weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.img_dim = getattr(args, 'img_dim', 256)
        self.val_percent = getattr(args, 'val', 10.0) / 100
        self.optimizer_name = getattr(args, 'optimizer', 'adamw').lower()  # Normalize to lowercase
        self.amp = getattr(args, 'amp', False)
        self.freeze_weights = getattr(args, 'freeze_weights', False)
        self.save_checkpoint = getattr(args, 'save_checkpoint', False)
        self.load_weights = getattr(args, 'load_weights', None)
        self.gradient_clipping = getattr(args, 'gradient_clipping', 1.0)
        self.dropout = getattr(args, 'dropout', 0.0)
        
        # Validate numeric parameters
        if self.batch_size <= 0:
            logging.warning(f"Invalid batch_size: {self.batch_size}. Using 16.")
            self.batch_size = 16
        if self.epochs <= 0:
            logging.warning(f"Invalid epochs: {self.epochs}. Using 50.")
            self.epochs = 50
        if self.learning_rate <= 0:
            logging.warning(f"Invalid learning_rate: {self.learning_rate}. Using 1e-4.")
            self.learning_rate = 1e-4
        if self.weight_decay < 0:
            logging.warning(f"Invalid weight_decay: {self.weight_decay}. Using 1e-4.")
            self.weight_decay = 1e-4
        if self.img_dim <= 0:
            logging.warning(f"Invalid img_dim: {self.img_dim}. Using 256.")
            self.img_dim = 256
        if self.val_percent <= 0 or self.val_percent >= 1:
            logging.warning(f"Invalid val_percent: {self.val_percent}. Using 0.1.")
            self.val_percent = 0.1
        if self.dropout < 0 or self.dropout > 1:
            logging.warning(f"Invalid dropout: {self.dropout}. Using 0.0.")
            self.dropout = 0.0
        
        # Validate optimizer
        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        if self.optimizer_name not in valid_optimizers:
            logging.warning(f"Invalid optimizer: {self.optimizer_name}. Using 'adamw'.")
            self.optimizer_name = 'adamw'
        
        # Workflow control parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.val_frequency = val_frequency
        self.project_name = project_name or self.model.__class__.__name__
        self.use_wandb = use_wandb
        self.experiment = None  # Will be set in setup_logging if wandb is used
        
        # Training state - will be initialized in setup methods
        self.metrics = {}           # Store metrics for tracking
        self.model_paths = {}       # Store paths to saved models
        self.early_stopping_counter = 0
        self.global_step = 0
        self.best_epoch = 0         # Track the epoch with the best performance
        
        # Set up data paths
        if self.dataset_dir is None:
            self.dir_img = Path('Dataset/Train/color')
            self.dir_mask = Path('Dataset/Train/label')
            self.dir_test_img = Path('Dataset/Test/color')
            self.dir_test_mask = Path('Dataset/Test/label')
        else:
            self.dir_img = self.dataset_dir / 'Train/color'
            self.dir_mask = self.dataset_dir / 'Train/label'
            self.dir_test_img = self.dataset_dir / 'Test/color'
            self.dir_test_mask = self.dataset_dir / 'Test/label'
        
        # Verify that required paths exist
        if not self.dir_img.exists():
            logging.warning(f"Training image directory not found: {self.dir_img}")
        if not self.dir_mask.exists():
            logging.warning(f"Training mask directory not found: {self.dir_mask}")
        
        logging.info(f"Trainer initialized for {self.model.__class__.__name__}")
        
        # Print training configuration at initialization
        config = get_training_config(self.model, self.args)
        config.update({
            "gradient_accumulation": self.gradient_accumulation_steps,
            "early_stopping": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_mode": self.early_stopping_mode,
            "val_frequency": self.val_frequency,
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else "default",
            "checkpoint_dir": str(self.checkpoint_dir)
        })
        
        print("\n" + "="*50)
        print(" "*10 + "TRAINING CONFIGURATION")
        print("="*50)
        
        for key, value in config.items():
            print(f"{key:>25}: {value}")
        
        print("="*50 + "\n")
    
    def setup(self):
        """Setup all components needed for training."""
        # Setup model on device
        self.model = self.model.to(self.device)
        
        try:
            # Setup training components
            self.setup_data()
            self.setup_training()
            self.setup_logging()
            self.model_specific_setup()
        except Exception as e:
            logging.error(f"Error during setup: {str(e)}")
            raise
    
    def setup_data(self):
        """Setup data loaders for training and validation."""
        try:
            # Check for test data
            self.has_test_set = self.dir_test_img.exists() and self.dir_test_mask.exists()
            if self.has_test_set:
                test_img_files = list(self.dir_test_img.glob('*'))
                test_mask_files = list(self.dir_test_mask.glob('*'))
                if test_img_files and test_mask_files:
                    logging.info(f"Found {len(test_img_files)} test images for final evaluation")
                else:
                    logging.warning("Test directories exist but no files found")
                    self.has_test_set = False
            
            # Setup data loaders using utility function
            self.train_loader, self.val_loader, self.train_set, self.val_set = setup_data_loaders(
                self.dir_img, self.dir_mask, self.batch_size, self.val_percent, 
                self.img_dim, test_mode=True
            )
            
            logging.info(f"Created dataset with {len(self.train_set)} training and {len(self.val_set)} validation images")
            
        except Exception as e:
            logging.error(f"Error setting up data: {str(e)}")
            raise
    
    def setup_training(self):
        """Set up training components (loss, optimizer, scheduler)."""
        try:
            # Setup loss function
            self.criterion = nn.CrossEntropyLoss(ignore_index=255) if self.model.n_classes > 1 else nn.BCEWithLogitsLoss()
            self.use_dice_loss = True
            
            # Freeze weights if requested
            if self.freeze_weights:
                freeze_model_layers(self.model, ratio=0.9, unfreeze_last_n=2)
                logging.info("Model weights frozen except for the last layers")
            
            # Setup optimizer and scheduler
            self.optimizer = setup_optimizer(
                self.model, self.optimizer_name, self.learning_rate, self.weight_decay
            )
            
            self.scheduler = setup_scheduler(
                self.optimizer, 'plateau', mode=self.early_stopping_mode, patience=5
            )
            
            # Setup mixed precision training
            self.grad_scaler = torch.amp.GradScaler(enabled=self.amp)
            
            # Initialize tracking variables
            self.best_val_score = float('-inf') if self.early_stopping_mode == 'max' else float('inf')
            self.best_val_score_after_epoch_10 = float('-inf') if self.early_stopping_mode == 'max' else float('inf')
            
        except Exception as e:
            logging.error(f"Error setting up training: {str(e)}")
            raise
    
    def setup_logging(self):
        """Set up experiment logging with wandb."""
        # Create config dictionary using the utility function
        config = get_training_config(self.model, self.args)
        
        # Add any trainer-specific configurations
        config.update({
            "gradient_accumulation": self.gradient_accumulation_steps,
            "early_stopping": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_mode": self.early_stopping_mode,
            "val_frequency": self.val_frequency,
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else "default",
            "checkpoint_dir": str(self.checkpoint_dir)
        })
        
        # Use the utility function to setup logging (without printing config again)
        self.experiment = setup_logging(config, self.project_name, skip_print=True, use_wandb=self.use_wandb)
    
    def model_specific_setup(self):
        """Model-specific setup steps. Override in subclasses if needed."""
        pass
    
    def get_batch_inputs(self, batch):
        """Extract model inputs from batch. Override in subclasses if needed."""
        return batch['image'].to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
    
    def get_batch_targets(self, batch):
        """Extract targets from batch. Override in subclasses if needed."""
        return batch['mask'].to(device=self.device, dtype=torch.long)
    
    def compute_loss(self, outputs, targets):
        """Compute loss between model outputs and targets."""
        loss = self.criterion(outputs, targets)
        
        # Add dice loss if enabled
        if self.use_dice_loss:
            loss += dice_loss(outputs, targets, n_classes=self.model.n_classes)
            
        return loss
    
    def log_metrics(self, metrics, loss, batch_idx, epoch, phase='train'):
        """
        Log metrics to wandb and console.
        
        Args:
            metrics: Dictionary containing metrics to log
            loss: Loss value for this batch
            batch_idx: Batch index
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
        """
        # Standardized metric names
        METRIC_NAMES = {
            'dice_scores': 'Dice',
            'iou_scores': 'IoU',
            'pixel_acc': 'Accuracy'
        }
        
        # Prepare metrics dict
        log_dict = {
            f'{phase}/loss': loss.item(),
        }
        
        # Add computed metrics if available using standardized names
        for metric_key, display_name in METRIC_NAMES.items():
            if metric_key in metrics:
                metric_value = metrics[metric_key]
                
                # Handle mean values
                if hasattr(metric_value, 'mean') and callable(metric_value.mean):
                    log_dict[f'{phase}/{display_name}/mean'] = metric_value.mean().item()
                    
                    # Add per-class metrics if this is a multi-class metric
                    if len(metric_value.shape) > 0 and metric_value.shape[0] > 1:
                        for i in range(metric_value.shape[0]):
                            log_dict[f'{phase}/{display_name}/class_{i}'] = metric_value[i].item()
                else:
                    # Single value metric
                    log_dict[f'{phase}/{display_name}'] = metric_value.item() if hasattr(metric_value, 'item') else metric_value
        
        # Add step/epoch
        if phase == 'train':
            log_dict['step'] = self.global_step
        log_dict['epoch'] = epoch
        
        # Log to wandb if available
        if self.experiment is not None:
            self.experiment.log(log_dict)
        
        # Store in metrics dict for access later
        for k, v in log_dict.items():
            self.metrics[k] = v
        
        # Always log to console for important metrics
        if batch_idx % 10 == 0:
            metrics_str = f"{phase.capitalize()} loss: {loss.item():.4f}"
            
            # Add key metrics to console output
            for metric_key, display_name in METRIC_NAMES.items():
                if metric_key in metrics:
                    metric_value = metrics[metric_key]
                    if hasattr(metric_value, 'mean') and callable(metric_value.mean):
                        metrics_str += f", {display_name}: {metric_value.mean().item():.4f}"
                    else:
                        metric_val = metric_value.item() if hasattr(metric_value, 'item') else metric_value
                        metrics_str += f", {display_name}: {metric_val:.4f}"
            
            logging.info(f"Epoch {epoch} - Batch {batch_idx}/{self.n_train // self.batch_size}: {metrics_str}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        # Track metrics
        epoch_loss = 0
        acc_loss = 0
        
        # Progress bar
        with torch.autocast(device_type=self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
            for batch_idx, batch in enumerate(self.train_loader):
                # Get batch data
                inputs = self.get_batch_inputs(batch)
                targets = self.get_batch_targets(batch)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling for mixed precision
                self.grad_scaler.scale(loss).backward()
                
                # Accumulate metrics
                acc_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update weights if enough accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clipping > 0:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    
                    # Update weights with gradient scaling
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Compute and log metrics
                    metrics = self.compute_metrics(outputs, targets)
                    self.log_metrics(metrics, torch.tensor(acc_loss), batch_idx, epoch)
                    
                    # Update tracking variables
                    epoch_loss += acc_loss
                    acc_loss = 0
                    self.global_step += 1
        
        # Report epoch stats
        epoch_loss /= len(self.train_loader)
        logging.info(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}")
    
    def validate(self, epoch):
        """
        Validate model on validation set.
        
        Args:
            epoch: Current epoch
            
        Returns:
            val_score: Validation score for early stopping
        """
        logging.info(f"Starting validation for epoch {epoch}...")
        try:
            self.model.eval()
            
            # Initialize metrics
            val_loss = 0
            num_batches = len(self.val_loader)
            
            # Run validation
            with torch.no_grad():
                # Use evaluate function from evaluate.py
                val_dice, val_iou, val_acc, val_dice_per_class, val_iou_per_class = evaluate(
                    self.model, self.val_loader, self.device, self.amp, 
                    dim=self.img_dim, n_classes=self.model.n_classes, 
                    desc=f'Validation (Epoch {epoch})'
                )
                
            val_metrics = {
                'dice_scores': val_dice_per_class,
                'iou_scores': val_iou_per_class,
                'pixel_acc': val_acc
            }
            
            # Compute average validation loss (placeholder since we calculate metrics differently)
            val_loss /= max(num_batches, 1)  # Avoid division by zero
            
            # Log validation metrics
            self.log_metrics(val_metrics, torch.tensor(val_loss), 0, epoch, phase='val')
            
            # Get the score for early stopping based on the specified metric
            if self.early_stopping_metric == 'val_dice':
                val_score = val_dice
                metric_name = "Dice"
            elif self.early_stopping_metric == 'val_iou':
                val_score = val_iou
                metric_name = "IoU"
            elif self.early_stopping_metric == 'val_accuracy':
                val_score = val_acc
                metric_name = "Accuracy"
            else:
                logging.warning(f"Unknown early stopping metric: {self.early_stopping_metric}. Using IoU.")
                val_score = val_iou
                metric_name = "IoU"
                
            # Log validation results
            logging.info(f"Epoch {epoch} - Validation Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Accuracy: {val_acc:.4f}")
            
            # Determine if this is the best model so far
            is_best = False
            
            if self.early_stopping_mode == 'max':
                is_best = val_score > self.best_val_score
            else:  # min mode
                is_best = val_score < self.best_val_score
                
            # Calculate score improvement for logging
            if epoch > 0:
                if self.early_stopping_mode == 'max':
                    improvement = val_score - self.best_val_score if is_best else 0
                    if is_best:
                        logging.info(f"Validation {metric_name} improved by {improvement:.4f}")
                else:
                    improvement = self.best_val_score - val_score if is_best else 0
                    if is_best:
                        logging.info(f"Validation {metric_name} decreased by {improvement:.4f}")
                    
            # Update best score if this is the best model
            if is_best:
                self.best_val_score = val_score
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                
                # Save best model
                self.save_best_model(epoch, val_score, prefix="best_model")
                logging.info(f"New best model saved with validation {metric_name}: {val_score:.4f}")
            else:
                self.early_stopping_counter += 1
                epochs_left = self.early_stopping_patience - self.early_stopping_counter
                logging.info(f"Validation {metric_name} did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                if epochs_left > 0:
                    logging.info(f"Training will stop in {epochs_left} epochs if no improvement occurs")
            
            # Use a second threshold after epoch 10 to catch models that improve later
            if epoch >= 10:
                if ((self.early_stopping_mode == 'max' and val_score > self.best_val_score_after_epoch_10) or 
                    (self.early_stopping_mode == 'min' and val_score < self.best_val_score_after_epoch_10)):
                    self.best_val_score_after_epoch_10 = val_score
                    # Save a separate best model checkpoint for after epoch 10
                    self.save_best_model(epoch, val_score, prefix="best_model_later_epochs")
                
            return val_score
            
        except Exception as e:
            logging.error(f"Error during validation: {str(e)}")
            logging.error(traceback.format_exc())
            return -float('inf') if self.early_stopping_mode == 'max' else float('inf')
    
    def save_best_model(self, epoch, score, prefix="best_model"):
        """
        Save the current model as the best model so far.
        
        Args:
            epoch: Current epoch
            score: Validation score
            prefix: Prefix for the checkpoint filename
        """
        checkpoint_path = save_checkpoint(
            self.model, self.optimizer, epoch, self.checkpoint_dir,
            self.metrics, True, f"{prefix}_epoch{epoch}.pth"
        )
        
        if checkpoint_path:
            # Store the path for later reference
            self.model_paths[prefix] = checkpoint_path
            logging.info(f"Saved {prefix} model at epoch {epoch} with score {score:.4f}")
        else:
            logging.error(f"Failed to save {prefix} model at epoch {epoch}")
            # Try one more time with a simplified name in current directory
            emergency_path = f"./{prefix}_emergency_epoch{epoch}.pth"
            try:
                torch.save({'model_state_dict': self.model.state_dict()}, emergency_path)
                self.model_paths[prefix] = emergency_path
                logging.warning(f"Saved emergency checkpoint to {emergency_path}")
            except Exception as e:
                logging.error(f"Failed to save emergency checkpoint: {str(e)}")
                # Keep old path if we have one
                if prefix in self.model_paths:
                    logging.info(f"Keeping previous best model: {self.model_paths[prefix]}")
        
        return self.model_paths.get(prefix, None)
            
    def train(self):
        """
        Train the model according to specified parameters.
        
        Returns:
            model: Trained model
            model_paths: Paths to saved model checkpoints
        """
        logging.info(f"Starting training for {self.epochs} epochs...")
        try:
            # Setup training components
            self.setup()
            
            # Load pretrained weights if specified
            if self.load_weights:
                if load_model_weights(self.model, self.load_weights, self.device):
                    logging.info(f"Successfully loaded pretrained weights from {self.load_weights}")
                else:
                    logging.error(f"Failed to load pretrained weights from {self.load_weights}")
            
            # Initialize validation tracking
            self.best_val_score = float('-inf') if self.early_stopping_mode == 'max' else float('inf')
            self.best_val_score_after_epoch_10 = float('-inf') if self.early_stopping_mode == 'max' else float('inf')
            self.early_stopping_counter = 0
            
            # Track training start time
            start_time = time.time()
            
            # Track for each epoch
            for epoch in range(1, self.epochs + 1):
                # Train for one epoch
                epoch_loss = self.train_epoch(epoch)
                
                # Validate model
                if epoch % self.val_frequency == 0:
                    val_score = self.validate(epoch)
                    
                    # Step the scheduler based on validation performance
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if self.early_stopping_mode == 'min':
                            self.scheduler.step(val_score)
                        else:
                            self.scheduler.step(-val_score)  # Negate score for 'max' mode
                    else:
                        self.scheduler.step()
                    
                    # Check for early stopping
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logging.info(f"Early stopping after {epoch} epochs. Best epoch was {self.best_epoch}.")
                        # Load the best model weights before returning
                        best_model_path = self.model_paths.get('best_model', None)
                        if best_model_path and os.path.exists(best_model_path):
                            load_model_weights(self.model, best_model_path, self.device)
                            logging.info(f"Loaded best model weights from epoch {self.best_epoch}")
                        break
                else:
                    # Step the scheduler for non-plateau schedulers when not validating
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
                
                # Save regular checkpoints if requested
                if self.save_checkpoint:
                    save_checkpoint(
                        self.model, self.optimizer, epoch, self.checkpoint_dir, 
                        self.metrics, True, f"checkpoint_epoch{epoch}.pth"
                    )
            
            # Final evaluation on test set
            if self.has_test_set:
                self.evaluate_test_set()
            
            # Log total training time
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logging.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            logging.info(f"Best model saved at epoch {self.best_epoch}")
            
            return self.model, self.model_paths
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def evaluate_test_set(self):
        """
        Evaluate the model on the test set.
        
        Returns:
            Dictionary of test results or None if evaluation failed
        """
        logging.info("Evaluating model on test set...")
        
        # Skip if no test set is available
        if not self.has_test_set:
            logging.warning("No test set found. Skipping test evaluation.")
            return None
        
        try:
            # Verify test directories exist and contain files
            if not self.dir_test_img.exists() or not self.dir_test_mask.exists():
                logging.error(f"Test directories not found: {self.dir_test_img} or {self.dir_test_mask}")
                return None
            
            test_img_files = list(self.dir_test_img.glob('*'))
            test_mask_files = list(self.dir_test_mask.glob('*'))
            
            if not test_img_files or not test_mask_files:
                logging.error(f"No test files found in {self.dir_test_img} or {self.dir_test_mask}")
                return None
            
            # Find best model checkpoint
            model_path = get_best_checkpoint_path(self.checkpoint_dir, self.model_paths)
            if not model_path:
                logging.warning("No model checkpoint found. Using current model weights.")
                model_path = "current_weights"
            else:
                # Load best model weights
                load_success = load_model_weights(self.model, model_path, self.device)
                if not load_success:
                    logging.warning("Failed to load best model weights. Using current weights.")
                    model_path = "current_weights"
            
            # Set appropriate model mode for autoencoder models
            if hasattr(self.model, 'set_segmentation_mode'):
                try:
                    self.model.set_segmentation_mode()
                    logging.info("Model set to segmentation mode for evaluation")
                except Exception as e:
                    logging.error(f"Failed to set model to segmentation mode: {str(e)}")
            
            # Create test dataset
            try:
                from data_loading import TestSegmentationDataset
                test_dataset = TestSegmentationDataset(
                    test_img_files, 
                    test_mask_files, 
                    dim=self.img_dim
                )
                
                test_loader = DataLoader(
                    test_dataset, batch_size=1, shuffle=False, 
                    num_workers=os.cpu_count(), pin_memory=True
                )
                logging.info(f"Created test dataset with {len(test_dataset)} images")
            except Exception as e:
                logging.error(f"Failed to create test dataset: {str(e)}")
                logging.error(traceback.format_exc())
                return None
            
            # Run evaluation
            try:
                test_results = evaluate(
                    self.model, test_loader, self.device, self.amp, 
                    dim=self.img_dim, n_classes=self.model.n_classes, 
                    desc='Test evaluation'
                )
                
                # Format results
                results = {
                    'dice': test_results[0],
                    'iou': test_results[1],
                    'acc': test_results[2],  # Changed from 'pixel_acc' to 'acc' to match utils/metrics.py
                    'dice_per_class': test_results[3],
                    'iou_per_class': test_results[4]
                }
            except Exception as e:
                logging.error(f"Error during test evaluation: {str(e)}")
                logging.error(traceback.format_exc())
                return None
            
            # Log results
            try:
                log_test_results(
                    self.checkpoint_dir, model_path, results, 
                    self.experiment, self.metrics
                )
                logging.info("Test evaluation complete")
            except Exception as e:
                logging.error(f"Error logging test results: {str(e)}")
                # Still return results even if logging failed
            
            return results
            
        except Exception as e:
            logging.error(f"Error during test evaluation: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def compute_metrics(self, outputs, targets):
        """Compute evaluation metrics for outputs and targets."""
        # Process targets to handle void label
        targets_processed = targets.clone()
        if targets.max() == 255:  # Check for void label
            targets_processed[targets_processed == 255] = 0
            
        # Get predictions based on model type
        if self.model.n_classes == 1:  # Binary segmentation
            pred = (outputs > 0).float()
        else:  # Multi-class segmentation
            pred = outputs.argmax(dim=1)
            
        # Compute metrics using functions from evaluate
        dice_scores = compute_dice_per_class(pred, targets_processed, n_classes=self.model.n_classes)
        iou_scores = compute_iou_per_class(pred, targets_processed, n_classes=self.model.n_classes)
        pixel_acc = compute_pixel_accuracy(pred, targets_processed)
        
        return {
            'dice_scores': dice_scores,
            'iou_scores': iou_scores,
            'pixel_acc': pixel_acc
        } 