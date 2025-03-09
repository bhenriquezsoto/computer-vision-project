"""
Utility functions for model training.
Provides common functions used by trainers for setup, data handling, and workflow management.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def setup_data_loaders(dir_img, dir_mask, batch_size, val_percent, img_dim, test_mode=False):
    """
    Set up training and validation data loaders.
    
    Args:
        dir_img: Directory containing images
        dir_mask: Directory containing masks
        batch_size: Batch size for training
        val_percent: Percentage of data to use for validation
        img_dim: Image dimension for preprocessing
        test_mode: Whether to use TestSegmentationDataset for validation
        
    Returns:
        train_loader, val_loader, train_set, val_set
    """
    from data_loading import SegmentationDataset, TestSegmentationDataset
    
    # Verify directories exist
    if not dir_img.exists() or not dir_mask.exists():
        raise FileNotFoundError(f"Data directories not found. Looked in {dir_img} and {dir_mask}")
    
    # Get image and mask files
    img_files = list(dir_img.glob('*'))
    mask_files = list(dir_mask.glob('*'))
    
    # Verify files are found
    if len(img_files) == 0 or len(mask_files) == 0:
        raise FileNotFoundError(f"No image or mask files found in {dir_img} and {dir_mask}")
    
    # Split into train and validation
    train_images, val_images, train_masks, val_masks = train_test_split(
        img_files, mask_files, test_size=val_percent, random_state=42
    )
    
    # Create datasets
    train_set = SegmentationDataset(train_images, train_masks, dim=img_dim)
    val_set_class = TestSegmentationDataset if test_mode else SegmentationDataset
    val_set = val_set_class(val_images, val_masks, dim=img_dim)
    
    # Create data loaders
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=batch_size, **loader_args
    )
    val_loader = DataLoader(
        val_set, shuffle=False, drop_last=True, batch_size=1, **loader_args
    )
    
    return train_loader, val_loader, train_set, val_set

def setup_optimizer(model, optimizer_name, learning_rate, weight_decay, only_trainable=True):
    """
    Set up the optimizer.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer to use
        learning_rate: Learning rate
        weight_decay: Weight decay
        only_trainable: Whether to only optimize trainable parameters
        
    Returns:
        The configured optimizer
    """
    # Get trainable parameters
    if only_trainable:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()
    
    # Create the optimizer
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        logging.warning(f"Unknown optimizer {optimizer_name}, using AdamW")
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

def setup_scheduler(optimizer, scheduler_name='plateau', mode='max', patience=5):
    """
    Set up a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_name: Name of the scheduler to use
        mode: 'min' or 'max' for PlateauLR
        patience: Patience for PlateauLR
        
    Returns:
        The configured scheduler
    """
    if scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.1, patience=patience, verbose=True
        )
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        logging.warning(f"Unknown scheduler {scheduler_name}, using ReduceLROnPlateau")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.1, patience=patience, verbose=True
        )

def freeze_model_layers(model, layers_to_freeze=None, ratio=None, unfreeze_last_n=0):
    """
    Freeze layers in a model.
    
    Args:
        model: The model to freeze layers in
        layers_to_freeze: List of layer names to freeze
        ratio: Ratio of parameters to freeze (from start)
        unfreeze_last_n: Number of last layers to keep unfrozen
        
    Returns:
        Number of frozen parameters
    """
    if layers_to_freeze:
        # Freeze specific layers by name
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                frozen_count += 1
        return frozen_count
    
    elif ratio and ratio > 0:
        # Freeze a ratio of parameters from the start
        params = list(model.parameters())
        num_to_freeze = int(len(params) * ratio)
        
        if unfreeze_last_n > 0:
            num_to_freeze = max(0, num_to_freeze - unfreeze_last_n)
            
        for i, param in enumerate(params):
            param.requires_grad = (i >= num_to_freeze)
        
        return num_to_freeze
    
    return 0  # No layers frozen 

def setup_logging(config, project_name="SegmentationProject", skip_print=False, use_wandb=True):
    """
    Set up experiment logging with wandb and print configuration.
    
    Args:
        config: Dictionary containing configuration parameters
        project_name: Name of the project for wandb
        skip_print: If True, skip printing the configuration (useful when config is printed elsewhere)
        use_wandb: If False, skip wandb initialization
        
    Returns:
        The initialized wandb experiment or None if wandb is not used
    """
    # Print configuration
    if not skip_print:
        print("\n" + "="*50)
        print(" "*10 + "TRAINING CONFIGURATION")
        print("="*50)
        
        for key, value in config.items():
            print(f"{key:>25}: {value}")
        
        print("="*50 + "\n")
    
    # Initialize wandb if requested
    experiment = None
    if use_wandb:
        try:
            # Use the global wandb import instead of reimporting it here
            experiment = wandb.init(
                project=project_name,
                config=config
            )
            logging.info(f"Started wandb logging with project: {project_name}")
        except ImportError:
            logging.warning("Could not import wandb. Running without wandb logging.")
        except Exception as e:
            logging.warning(f"Error initializing wandb: {str(e)}. Running without wandb logging.")
    
    return experiment

def get_training_config(model, args):
    """
    Create a configuration dictionary from model and arguments.
    
    Args:
        model: The model to train
        args: Training arguments
        
    Returns:
        Dictionary with configuration parameters
    """
    # Extract n_classes from model if available
    n_classes = getattr(model, 'n_classes', None)
    if n_classes is None:
        # Try to infer from output layers
        if hasattr(model, 'outc') and hasattr(model.outc, 'out_channels'):
            n_classes = model.outc.out_channels
        else:
            n_classes = 1  # Default if can't determine
    
    # Get learning rate - handle both 'lr' and 'learning_rate' for compatibility
    learning_rate = getattr(args, 'lr', getattr(args, 'learning_rate', 1e-4))
    
    config = {
        "model": model.__class__.__name__,
        "epochs": getattr(args, 'epochs', 0),
        "batch_size": getattr(args, 'batch_size', 0),
        "learning_rate": learning_rate,
        "weight_decay": getattr(args, 'weight_decay', 0),
        "optimizer": getattr(args, 'optimizer', 'adamw'),
        "image_dim": getattr(args, 'img_dim', 0),
        "gradient_accumulation": getattr(args, 'gradient_accumulation_steps', 1),
        "early_stopping": getattr(args, 'early_stopping_patience', 0),
        "grad_clipping": getattr(args, 'gradient_clipping', 0),
        "amp": getattr(args, 'amp', False),
        "classes": n_classes,
    }
    
    # Add custom fields if present
    if hasattr(args, 'scheduler'):
        config["scheduler"] = args.scheduler
    
    if hasattr(args, 'dropout'):
        config["dropout"] = args.dropout
    
    return config 