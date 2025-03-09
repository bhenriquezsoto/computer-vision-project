#!/usr/bin/env python3
"""
Main training script for segmentation models.
This script dispatches to model-specific trainers.
"""
import argparse
import logging
import os
from pathlib import Path
import torch

# Import model classes
from models.unet.unet_model import UNet
from models.clip.clip_model import CLIPSegmentationModel
from models.autoencoder.auto_encoder_model import AutoencoderSegmentation

# Import trainer classes
from models.unet.unet_trainer import UNetTrainer
from models.clip.clip_trainer import CLIPTrainer
from models.autoencoder.autoencoder_trainer import AutoencoderTrainer
from models.base.registry import _MODEL_TRAINER_REGISTRY, register_model_trainer

# Manually register trainers
_MODEL_TRAINER_REGISTRY['UNet'] = UNetTrainer
_MODEL_TRAINER_REGISTRY['CLIPSegmentationModel'] = CLIPTrainer
_MODEL_TRAINER_REGISTRY['AutoencoderSegmentation'] = AutoencoderTrainer

# Global constants
CHECKPOINT_DIR = Path('src/models/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Map of model types to their specialized parameters
MODEL_PARAMS = {
    'autoencoder': {
        'model_class': AutoencoderSegmentation,
        'specialized_params': [
            'mode', 'epochs_recon', 'epochs_seg', 'freeze_ratio', 'load_encoder'
        ]
    },
    'unet': {
        'model_class': UNet,
        'specialized_params': ['bilinear']
    },
    'clip': {
        'model_class': CLIPSegmentationModel,
        'specialized_params': ['train_clip_backbone']
    }
}

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_model(args):
    """Create a model based on the arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        model: The model instance
    """
    try:
        model_info = MODEL_PARAMS.get(args.model)
        if not model_info:
            raise ValueError(f"Unsupported model: {args.model}")
            
        model_class = model_info['model_class']
        
        if args.model == 'unet':
            model = model_class(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        elif args.model == 'clip':
            model = model_class(n_classes=args.classes)
        elif args.model == 'autoencoder':
            model = model_class(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
            
        # Apply dropout if specified
        if args.dropout > 0:
            import functools
            import torch.nn.functional as F
            
            def dropout_hook(module, input, output, p=0.0):
                """Hook function to apply dropout to layer outputs"""
                return F.dropout(output, p=p, training=module.training)
            
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    module.register_forward_hook(
                        functools.partial(dropout_hook, p=args.dropout)
                    )
            logging.info(f"Applied dropout with probability {args.dropout} to all Conv2d and Linear layers")
        
        # Load pretrained weights if specified
        if args.load_weights:
            try:
                state_dict = torch.load(args.load_weights, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Try to load weights, allowing for partial matches
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logging.warning(f"Missing keys when loading weights: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys in weight file: {unexpected_keys}")
                    
                logging.info(f"Loaded pretrained weights from {args.load_weights}")
            except Exception as e:
                logging.error(f"Failed to load weights: {str(e)}")
                raise
        
        model = model.to(memory_format=torch.channels_last)
        return model
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}")
        raise

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train segmentation models')
    
    # Common arguments for all models
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'clip', 'autoencoder'], default='unet',
                        help='Choose model (unet, clip, or autoencoder)')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', '-o', type=str, choices=['adamw', 'adam', 'rmsprop', 'sgd'], default='adamw', 
                       help='Choose optimizer')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--dataset-dir', type=str, default=None,
                       help='Path to dataset directory containing Train and Test folders')
    
    # Enhanced training parameters
    parser.add_argument('--gradient-accumulation', type=int, default=1, 
                       help='Number of gradient accumulation steps')
    parser.add_argument('--early-stopping', type=int, default=10, 
                       help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--val-frequency', type=int, default=1,
                      help='Validate every N epochs')
    
    # Weight and dropout arguments (work with all models)
    parser.add_argument('--dropout', type=float, default=0.0, 
                       help='Dropout probability (0 to disable, applies to all models)')
    parser.add_argument('--load-weights', type=str, default=None, 
                       help='Load pretrained weights (works with all models)')
    parser.add_argument('--freeze-weights', action='store_true', 
                       help='Freeze pretrained weights except last layer (works with all models)')
    parser.add_argument('--save-checkpoint', action='store_true', default=False,
                      help='Save checkpoint after each epoch')
    
    # UNet specific arguments
    parser.add_argument('--bilinear', action='store_true', default=False, 
                       help='[UNet] Use bilinear upsampling instead of transposed convolutions')
    
    # CLIP specific arguments
    parser.add_argument('--train-clip-backbone', action='store_true', default=False,
                      help='[CLIP] Train the CLIP backbone (otherwise frozen)')
    
    # Autoencoder specific arguments
    parser.add_argument('--mode', type=str, choices=['reconstruction', 'segmentation', 'both'], 
                       default='both', help='[Autoencoder] Training mode')
    parser.add_argument('--epochs-recon', type=int, default=40, 
                       help='[Autoencoder] Number of epochs for reconstruction phase')
    parser.add_argument('--epochs-seg', type=int, default=40, 
                       help='[Autoencoder] Number of epochs for segmentation phase')
    parser.add_argument('--freeze-ratio', type=float, default=0.8, 
                       help='[Autoencoder] Ratio of encoder layers to freeze in segmentation phase')
    parser.add_argument('--load-encoder', type=str, default=None, 
                       help='[Autoencoder] Path to pretrained encoder weights for segmentation phase')
    
    # Logging and monitoring
    parser.add_argument('--no-wandb', action='store_true', default=False,
                      help='Disable wandb logging')
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = get_args()
    
    # Setup logging
    setup_logging()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    try:
        # Create model
        model = get_model(args)
        
        # Get the appropriate trainer for this model
        from models.base.registry import get_trainer_for_model, _MODEL_TRAINER_REGISTRY
        logging.info(f"Registry state at import: {list(_MODEL_TRAINER_REGISTRY.keys())}")
        model_class_name = model.__class__.__name__
        logging.info(f"Looking up trainer for model class: {model_class_name}")
        trainer_class = get_trainer_for_model(model_class_name)
        
        # Create directory for model checkpoints
        checkpoint_dir = CHECKPOINT_DIR / args.model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Build trainer parameters
        trainer_params = {
            'model': model,
            'args': args,
            'device': device,
            'dataset_dir': args.dataset_dir,
            'checkpoint_dir': checkpoint_dir,
            'gradient_accumulation_steps': args.gradient_accumulation,
            'early_stopping_patience': args.early_stopping,
            'val_frequency': args.val_frequency,
            'use_wandb': not args.no_wandb
        }
        
        # Note: We don't need to explicitly add 'bilinear' to trainer_params
        # because UNetTrainer gets it from the args object using getattr(args, 'bilinear', False)
        
        # Add model-specific parameters if available - but skip 'bilinear' which is handled specially
        model_info = MODEL_PARAMS.get(args.model, {})
        for param in model_info.get('specialized_params', []):
            if param != 'bilinear' and hasattr(args, param):
                trainer_params[param] = getattr(args, param)
        
        # Create trainer
        trainer = trainer_class(**trainer_params)
        
        # Train the model
        model, model_paths = trainer.train()
        logging.info(f"Training complete. Model saved at: {list(model_paths.values())}")
        
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                    'Enabling checkpointing to reduce memory usage, but this slows down training. '
                    'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        
        # Try again with checkpointing
        model = get_model(args)
        model.use_checkpointing()
        
        # Get the appropriate trainer for this model
        from models.base.registry import get_trainer_for_model
        trainer_class = get_trainer_for_model(model.__class__.__name__)
        
        # Create trainer instance with checkpointing enabled
        trainer = trainer_class(
            model=model,
            args=args,
            device=device,
            dataset_dir=args.dataset_dir,
            checkpoint_dir=CHECKPOINT_DIR / args.model,
            gradient_accumulation_steps=args.gradient_accumulation,
            early_stopping_patience=args.early_stopping,
            val_frequency=args.val_frequency
        )
        
        # Train the model with checkpointing
        model, model_paths = trainer.train()
        logging.info(f"Training complete with checkpointing. Model saved at: {list(model_paths.values())}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
