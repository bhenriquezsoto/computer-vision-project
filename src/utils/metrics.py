"""
Metrics and evaluation utilities for segmentation models.
This module provides functions for computing, tracking, and saving 
metrics for image segmentation models.
"""
import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Try to import wandb, but don't fail if it's not available
try:
    import wandb
except ImportError:
    logging.warning("wandb not found. Some logging features will be disabled.")
    wandb = None

def get_best_checkpoint_path(checkpoint_dir, model_paths=None):
    """
    Find the best model checkpoint path.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_paths: Dictionary mapping checkpoint type to path
        
    Returns:
        Path to the best model checkpoint
    """
    if model_paths and 'best_model_after_epoch_10' in model_paths:
        return model_paths['best_model_after_epoch_10']
    elif model_paths and 'best_model' in model_paths:
        return model_paths['best_model']
    else:
        # If no best model was saved, use the latest checkpoint
        checkpoint_files = list(Path(checkpoint_dir).glob('checkpoint_epoch*.pth'))
        if not checkpoint_files:
            return None
            
        latest_epoch = max([int(f.stem.replace('checkpoint_epoch', '')) 
                          for f in checkpoint_files])
        return os.path.join(checkpoint_dir, f'checkpoint_epoch{latest_epoch}.pth')

def load_model_weights(model, model_path, device):
    """
    Load model weights from checkpoint.
    
    Args:
        model: The model to load weights into
        model_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        True if weights were loaded successfully, False otherwise
    """
    if not model:
        logging.error("No model provided for loading weights")
        return False
        
    if not model_path:
        logging.error("No model path provided for loading weights")
        return False
        
    if not os.path.exists(model_path):
        logging.error(f"Model checkpoint not found: {model_path}")
        return False
        
    try:
        # Load the state dict from file
        try:
            state_dict = torch.load(model_path, map_location=device)
            logging.info(f"Successfully loaded checkpoint file: {model_path}")
        except (RuntimeError, ValueError) as e:
            if 'unexpected EOF' in str(e).lower() or 'corrupt' in str(e).lower():
                logging.error(f"Checkpoint file is corrupted: {model_path}")
            else:
                logging.error(f"Error loading checkpoint file: {str(e)}")
            return False
        except FileNotFoundError:
            logging.error(f"Checkpoint file not found: {model_path}")
            return False
        except Exception as e:
            logging.error(f"Unknown error loading checkpoint: {str(e)}")
            return False
            
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                logging.info("Found 'model_state_dict' key in checkpoint")
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                logging.info("Found 'state_dict' key in checkpoint")
            elif 'model' in state_dict:
                state_dict = state_dict['model']
                logging.info("Found 'model' key in checkpoint")
            elif not any(k.startswith('module.') or k.startswith('encoder.') or k.startswith('decoder.') for k in state_dict.keys()):
                # This doesn't look like a model state dict
                logging.warning("Checkpoint format doesn't look like a model state dict")
                # Try to find any state dict-like keys
                possible_sd_keys = [k for k in state_dict.keys() if isinstance(state_dict[k], dict) and any(
                    isinstance(state_dict[k][sub_k], torch.Tensor) for sub_k in state_dict[k] if isinstance(sub_k, str)
                )]
                if possible_sd_keys:
                    logging.info(f"Found potential state dict keys: {possible_sd_keys}")
                    state_dict = state_dict[possible_sd_keys[0]]
                    logging.info(f"Using key '{possible_sd_keys[0]}' as state dict")
            
        # Try to load the state dict
        try:
            # First try direct loading
            model.load_state_dict(state_dict)
            logging.info(f'Model weights loaded successfully from {model_path}')
            return True
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            # Check for common errors
            if 'size mismatch' in error_msg:
                logging.error(f"Model architecture mismatch - size mismatch: {str(e)}")
                logging.error("The model checkpoint architecture doesn't match the current model.")
                
                # Try to provide more details about the mismatch
                if hasattr(model, 'n_classes'):
                    logging.error(f"Current model has {model.n_classes} output classes.")
                if hasattr(model, 'n_channels'):
                    logging.error(f"Current model has {model.n_channels} input channels.")
                    
            elif 'missing key(s)' in error_msg:
                logging.error(f"Model architecture mismatch - missing keys: {str(e)}")
                logging.error("The checkpoint is missing some weights needed by the current model.")
                
            elif 'unexpected key(s)' in error_msg:
                logging.error(f"Model architecture mismatch - unexpected keys: {str(e)}")
                logging.error("The checkpoint contains weights not used by the current model.")
                
            else:
                logging.error(f"Failed to load model weights: {str(e)}")
                
            # Try using strict=False as a fallback
            try:
                logging.info("Attempting to load weights with strict=False...")
                model.load_state_dict(state_dict, strict=False)
                logging.warning("Model weights loaded with strict=False (some weights may be missing or unused)")
                return True
            except Exception as e2:
                logging.error(f"Failed to load weights even with strict=False: {str(e2)}")
                
            return False
        except Exception as e:
            logging.error(f"Unexpected error loading model weights: {str(e)}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to load model weights: {str(e)}")
        return False

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, metrics=None, add_metrics=True, filename=None):
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        epoch: Current epoch
        checkpoint_dir: Directory to save the checkpoint
        metrics: Dictionary of metrics to save
        add_metrics: Whether to add metrics to the checkpoint
        filename: Filename to save as (default: checkpoint_epoch{epoch}.pth)
        
    Returns:
        Path to the saved checkpoint
    """
    # Ensure model and optimizer are provided
    if model is None:
        logging.error("Cannot save checkpoint: model is None")
        return None
    if optimizer is None:
        logging.warning("Saving checkpoint without optimizer state")
    
    # Create checkpoint directory if it doesn't exist
    try:
        # Convert to Path object for better OS compatibility
        checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create checkpoint directory {checkpoint_dir}: {str(e)}")
        # Try to save in current directory as fallback
        checkpoint_dir = Path('.')
        logging.warning(f"Using current directory as fallback")
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    # Add optimizer state if available
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add metrics if requested
    if add_metrics and metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Determine filename
    if filename is None:
        filename = f'checkpoint_epoch{epoch}.pth'
    
    # Create full path
    try:
        checkpoint_path = checkpoint_dir / filename
    except Exception as e:
        logging.error(f"Failed to create checkpoint path: {str(e)}")
        checkpoint_path = Path(f"./emergency_checkpoint_{epoch}.pth")
        logging.warning(f"Using emergency filename: {checkpoint_path}")
    
    # Save checkpoint
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {str(e)}")
        return None

def log_test_results(output_dir, model_path, test_results, experiment=None, metrics_dict=None):
    """
    Log test results to file, console, and wandb.
    
    Args:
        output_dir: Directory to save results file
        model_path: Path to the model that was evaluated
        test_results: Dictionary containing test results
        experiment: WandB experiment for logging (or None if wandb not used)
        metrics_dict: Dictionary to update with metrics
    """
    # Validate inputs
    if not test_results:
        logging.error("No test results provided for logging")
        return
    
    if not isinstance(test_results, dict):
        logging.error(f"Expected test_results to be a dictionary, got {type(test_results)}")
        return
    
    # Extract metrics with standard names
    try:
        test_dice = test_results.get('dice', 0.0)
        test_iou = test_results.get('iou', 0.0)
        test_acc = test_results.get('acc', 0.0)
        test_dice_per_class = test_results.get('dice_per_class', torch.zeros(1))
        test_iou_per_class = test_results.get('iou_per_class', torch.zeros(1))
        n_classes = len(test_dice_per_class)
    except Exception as e:
        logging.error(f"Error extracting metrics from test_results: {str(e)}")
        return
    
    # Create output directory if it doesn't exist
    try:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
        # Use current directory as fallback
        output_dir = Path('.')
        logging.warning(f"Using current directory as fallback for test results")
    
    # Log results to file
    try:
        model_name = os.path.basename(model_path) if model_path else "unknown_model"
        result_file = output_dir / 'test_results.txt'
        
        with open(result_file, 'a') as f:
            f.write(f"\n\nTest results for model: {model_name}\n")
            f.write(f"Average Dice score: {test_dice:.4f}\n")
            f.write(f"Average IoU score: {test_iou:.4f}\n")
            f.write(f"Pixel accuracy: {test_acc:.4f}\n")
            f.write("Per-class metrics:\n")
            for i in range(n_classes):
                try:
                    dice_val = test_dice_per_class[i].item() if hasattr(test_dice_per_class[i], 'item') else test_dice_per_class[i]
                    iou_val = test_iou_per_class[i].item() if hasattr(test_iou_per_class[i], 'item') else test_iou_per_class[i]
                    f.write(f"  Class {i}: Dice={dice_val:.4f}, IoU={iou_val:.4f}\n")
                except IndexError:
                    f.write(f"  Class {i}: Data not available\n")
                except Exception as e:
                    f.write(f"  Class {i}: Error getting data: {str(e)}\n")
        
        logging.info(f"Test results written to {result_file}")
    except Exception as e:
        logging.error(f"Error writing test results to file: {str(e)}")
    
    # Log to console
    logging.info(f"Test results for model: {os.path.basename(model_path) if model_path else 'unknown_model'}")
    logging.info(f"Average Dice score: {test_dice:.4f}")
    logging.info(f"Average IoU score: {test_iou:.4f}")
    logging.info(f"Pixel accuracy: {test_acc:.4f}")
    logging.info("Per-class metrics:")
    for i in range(n_classes):
        try:
            dice_val = test_dice_per_class[i].item() if hasattr(test_dice_per_class[i], 'item') else test_dice_per_class[i]
            iou_val = test_iou_per_class[i].item() if hasattr(test_iou_per_class[i], 'item') else test_iou_per_class[i]
            logging.info(f"  Class {i}: Dice={dice_val:.4f}, IoU={iou_val:.4f}")
        except Exception as e:
            logging.info(f"  Class {i}: Error getting data: {str(e)}")
    
    # Log test results to wandb using standardized naming convention
    if experiment:
        try:
            test_metrics = {
                "test/Dice/mean": test_dice,
                "test/IoU/mean": test_iou,
                "test/Accuracy": test_acc,
            }
            
            # Add per-class metrics
            for i in range(n_classes):
                try:
                    dice_val = test_dice_per_class[i].item() if hasattr(test_dice_per_class[i], 'item') else test_dice_per_class[i]
                    iou_val = test_iou_per_class[i].item() if hasattr(test_iou_per_class[i], 'item') else test_iou_per_class[i]
                    test_metrics[f"test/Dice/class_{i}"] = dice_val
                    test_metrics[f"test/IoU/class_{i}"] = iou_val
                except Exception as e:
                    logging.warning(f"Error logging class {i} metrics to wandb: {str(e)}")
            
            # Log metrics
            experiment.log(test_metrics)
            logging.info("Test results logged to wandb")
            
            # Store in metrics dict if provided
            if metrics_dict is not None:
                for k, v in test_metrics.items():
                    metrics_dict[k] = v
        except Exception as e:
            logging.error(f"Error logging test results to wandb: {str(e)}")
    
    # Update metrics dict even if wandb is not used
    elif metrics_dict is not None:
        try:
            metrics_dict.update({
                "test/Dice/mean": test_dice,
                "test/IoU/mean": test_iou,
                "test/Accuracy": test_acc,
            })
            
            # Add per-class metrics
            for i in range(n_classes):
                try:
                    dice_val = test_dice_per_class[i].item() if hasattr(test_dice_per_class[i], 'item') else test_dice_per_class[i]
                    iou_val = test_iou_per_class[i].item() if hasattr(test_iou_per_class[i], 'item') else test_iou_per_class[i]
                    metrics_dict[f"test/Dice/class_{i}"] = dice_val
                    metrics_dict[f"test/IoU/class_{i}"] = iou_val
                except Exception as e:
                    logging.warning(f"Error adding class {i} metrics to metrics dict: {str(e)}")
        except Exception as e:
            logging.error(f"Error updating metrics dict: {str(e)}") 