import argparse
import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from metrics import compute_metrics
from models.unet_model import UNet
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder
from data_loading import TestSegmentationDataset, sort_and_match_files

# Set up directories
dir_test_img = Path('Dataset/Test/color')
dir_test_mask = Path('Dataset/Test/label')
dir_checkpoint = Path('src/models/checkpoints/')

def evaluate_model(
    model,
    device,
    img_dim=256,
    amp=False,
    n_classes=3,
    test_img_dir=dir_test_img,
    test_mask_dir=dir_test_mask,
    results_path=None,
    model_path=None,
    in_training=False
):
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: The model to evaluate
        device: Device to run evaluation on
        img_dim: Image dimension
        amp: Whether to use mixed precision
        n_classes: Number of classes
        test_img_dir: Directory with test images
        test_mask_dir: Directory with test masks
        results_path: Path to save results (if None, derived from model_path)
        model_path: Path of the model (used for generating results_path if needed)
        in_training: Whether the evaluation is in training phase
    Returns:
        Tuple of (mean_dice, mean_iou, mean_acc, dice_per_class, iou_per_class)
    """
    # Check if test directories exist
    if not test_img_dir.exists() or not test_mask_dir.exists():
        logging.error(f"Test directories not found. Please ensure these exist:")
        logging.error(f"- {test_img_dir}")
        logging.error(f"- {test_mask_dir}")
        return None, None, None, None, None
    
    # Get test files
    test_img_files = list(test_img_dir.glob('*'))
    test_mask_files = list(test_mask_dir.glob('*'))
    
    if len(test_img_files) == 0 or len(test_mask_files) == 0:
        logging.error(f"No files found in the test directories!")
        return None, None, None, None, None
    
    logging.info(f"Found {len(test_img_files)} test images and {len(test_mask_files)} test masks")
    
    # Sort and match test images and masks
    test_img_files, test_mask_files = sort_and_match_files(test_img_files, test_mask_files)
    
    # Create test dataset and dataloader
    test_dataset = TestSegmentationDataset(test_img_files, test_mask_files, dim=img_dim)
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_args)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Run evaluation using compute_metrics from metrics.py
    mean_dice, mean_iou, mean_acc, dice_per_class, iou_per_class = compute_metrics(
        model, test_loader, device, amp, dim=img_dim, n_classes=n_classes, desc='Testing round'
    )
    
    # Print results
    logging.info("=== Test Results ===")
    logging.info(f"Mean Dice Score: {mean_dice:.4f}")
    logging.info(f"Mean IoU: {mean_iou:.4f}")
    logging.info(f"Mean Pixel Accuracy: {mean_acc:.4f}")
    
    logging.info("Per-class metrics:")
    for i in range(n_classes):
        logging.info(f"Class {i}:")
        logging.info(f"  - Dice Score: {dice_per_class[i].item():.4f}")
        logging.info(f"  - IoU: {iou_per_class[i].item():.4f}")
    
    # Save results to file if path provided
    if results_path is None and model_path is not None:
        # Generate results path from model path
        if isinstance(model_path, str):
            results_path = model_path.replace('.pth', '_test_results.txt')
        else:
            results_path = str(model_path).replace('.pth', '_test_results.txt')
    
    if results_path:
        with open(results_path, 'w') as f:
            f.write(f"Test Results for model:\n")
            f.write(f"Mean Dice Score: {mean_dice:.4f}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            f.write(f"Mean Pixel Accuracy: {mean_acc:.4f}\n\n")
            
            f.write("Per-class metrics:\n")
            for i in range(n_classes):
                f.write(f"Class {i}:\n")
                f.write(f"  - Dice Score: {dice_per_class[i].item():.4f}\n")
                f.write(f"  - IoU: {iou_per_class[i].item():.4f}\n")
        
        if in_training:
            logging.info(f"Results saved to {results_path}")
    
    return mean_dice, mean_iou, mean_acc, dice_per_class, iou_per_class

def get_args():
    parser = argparse.ArgumentParser(description='Test a trained model on the test set')
    parser.add_argument('--model-path', '-m', type=str, required=True, help='Path to the saved model (.pth file)')
    parser.add_argument('--model-type', '-t', type=str, choices=['unet', 'clip', 'autoencoder'], default='unet', 
                       help='Type of model to test')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling (UNet only)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Testing model from {args.model_path}')
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Initialize model based on type
    if args.model_type == 'unet':
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_type == 'clip':
        model = CLIPSegmentationModel(n_classes=args.classes)
    elif args.model_type == 'autoencoder':
        model = Autoencoder(n_channels=3, n_classes=args.classes)
        # Ensure we're in segmentation phase for testing
        model.set_phase("segmentation")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Load saved model
    state_dict = torch.load(args.model_path, map_location=device)
    
    # Check if state_dict has model_state_dict key (our saving format)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    logging.info(f'Model loaded from {args.model_path}')
    model.to(device)
    
    # Use the evaluate_model function
    results_path = args.model_path.replace('.pth', '_test_results.txt')
    evaluate_model(
        model=model,
        device=device,
        img_dim=args.img_dim,
        amp=args.amp,
        n_classes=args.classes,
        results_path=results_path,
        model_path=args.model_path
    )

if __name__ == '__main__':
    main()