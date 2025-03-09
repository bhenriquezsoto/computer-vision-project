import argparse
import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from evaluate import evaluate

from models.unet_model import UNet
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder
from data_loading import TestSegmentationDataset

# Set up directories
dir_test_img = Path('Dataset/Test/color')
dir_test_mask = Path('Dataset/Test/label')
dir_checkpoint = Path('src/models/checkpoints/')

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
    model.eval()
    
    # Check if test directories exist
    if not dir_test_img.exists() or not dir_test_mask.exists():
        logging.error(f"Test directories not found. Please ensure these exist:")
        logging.error(f"- {dir_test_img}")
        logging.error(f"- {dir_test_mask}")
        return
    
    # Get test files
    test_img_files = list(dir_test_img.glob('*'))
    test_mask_files = list(dir_test_mask.glob('*'))
    
    if len(test_img_files) == 0 or len(test_mask_files) == 0:
        logging.error(f"No files found in the test directories!")
        return
    
    logging.info(f"Found {len(test_img_files)} test images and {len(test_mask_files)} test masks")
    
    # Create test dataset and dataloader
    test_dataset = TestSegmentationDataset(test_img_files, test_mask_files, dim=args.img_dim)
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_args)
    
    # Run evaluation directly
    mean_dice, mean_iou, mean_acc, dice_per_class, iou_per_class = evaluate(
        model, test_loader, device, args.amp, dim=args.img_dim, n_classes=args.classes, desc='Testing round'
    )
    
    # Print results
    logging.info("=== Test Results ===")
    logging.info(f"Mean Dice Score: {mean_dice:.4f}")
    logging.info(f"Mean IoU: {mean_iou:.4f}")
    logging.info(f"Mean Pixel Accuracy: {mean_acc:.4f}")
    
    logging.info("Per-class metrics:")
    for i in range(args.classes):
        logging.info(f"Class {i}:")
        logging.info(f"  - Dice Score: {dice_per_class[i].item():.4f}")
        logging.info(f"  - IoU: {iou_per_class[i].item():.4f}")
    
    # Save results to file
    results_path = args.model_path.replace('.pth', '_test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Results for {args.model_path}:\n")
        f.write(f"Mean Dice Score: {mean_dice:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Pixel Accuracy: {mean_acc:.4f}\n\n")
        
        f.write("Per-class metrics:\n")
        for i in range(args.classes):
            f.write(f"Class {i}:\n")
            f.write(f"  - Dice Score: {dice_per_class[i].item():.4f}\n")
            f.write(f"  - IoU: {iou_per_class[i].item():.4f}\n")
    
    logging.info(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()