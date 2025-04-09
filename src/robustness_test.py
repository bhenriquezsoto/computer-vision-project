import argparse
import logging
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
from tqdm import tqdm
import random
from skimage.util import random_noise
import torch.nn.functional as F

# Import from other project files
from metrics import compute_dice_per_class
from data_loading import TestSegmentationDataset, sort_and_match_files, load_image
from models.unet_model import UNet, PointUNet
from models.clip_model import CLIPSegmentationModel, CLIPUNet
from models.autoencoder_model import Autoencoder

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def add_gaussian_noise(image, std_dev):
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image as a numpy array
        std_dev: Standard deviation of the Gaussian noise
        
    Returns:
        Perturbed image with Gaussian noise
    """
    if std_dev == 0:
        return image.copy()
    
    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
    
    # Add noise to the image
    noisy_image = image.astype(np.float32) + noise
    
    # Clip to ensure valid pixel range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def apply_gaussian_blur(image, iterations):
    """
    Apply Gaussian blur to an image using the specified kernel.
    
    Args:
        image: Input image as a numpy array
        iterations: Number of times to apply the blur
        
    Returns:
        Blurred image
    """
    if iterations == 0:
        return image.copy()
    
    # Define the 3x3 Gaussian blur kernel (as shown in the image)
    kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ], dtype=np.float32)
    
    # Make a copy of the image to avoid modifying the original
    blurred_image = image.copy()
    
    # Apply the kernel multiple times
    for _ in range(iterations):
        blurred_image = cv2.filter2D(blurred_image, -1, kernel)
    
    return blurred_image.astype(np.uint8)

def adjust_contrast(image, factor):
    """
    Adjust the contrast of an image.
    
    Args:
        image: Input image as a numpy array
        factor: Contrast adjustment factor (>1 for increase, <1 for decrease)
        
    Returns:
        Contrast-adjusted image
    """
    # Convert to float for multiplication
    adjusted_image = image.astype(np.float32) * factor
    
    # Clip to ensure valid pixel range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image

def adjust_brightness(image, offset):
    """
    Adjust the brightness of an image.
    
    Args:
        image: Input image as a numpy array
        offset: Brightness adjustment value (positive for increase, negative for decrease)
        
    Returns:
        Brightness-adjusted image
    """
    # Add offset to pixel values
    adjusted_image = image.astype(np.float32) + offset
    
    # Clip to ensure valid pixel range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image

def apply_occlusion(image, box_size):
    """
    Apply occlusion to an image by placing a black square at a random position.
    
    Args:
        image: Input image as a numpy array
        box_size: Size of the square box to occlude
        
    Returns:
        Occluded image
    """
    if box_size == 0:
        return image.copy()
    
    # Make a copy of the image to avoid modifying the original
    occluded_image = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate valid positions for the box
    max_x = width - box_size
    max_y = height - box_size
    
    if max_x < 0 or max_y < 0:
        # Box is bigger than image, just return black image
        return np.zeros_like(image)
    
    # Generate random position for the box
    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)
    
    # Create occlusion (black box)
    if len(image.shape) == 3:  # Color image (H, W, C)
        occluded_image[y:y+box_size, x:x+box_size, :] = 0
    else:  # Grayscale image (H, W)
        occluded_image[y:y+box_size, x:x+box_size] = 0
    
    return occluded_image

def add_salt_pepper_noise(image, amount):
    """
    Add salt and pepper noise to an image.
    
    Args:
        image: Input image as a numpy array
        amount: Proportion of image pixels to replace with noise
        
    Returns:
        Image with salt and pepper noise
    """
    if amount == 0:
        return image.copy()
    
    # Use skimage's random_noise function to add salt and pepper noise
    noisy_image = random_noise(image, mode='s&p', amount=amount)
    
    # Convert back to uint8
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    return noisy_image

def apply_perturbation(image, perturbation_type, level, params):
    """
    Apply a specified perturbation to an image.
    
    Args:
        image: Input image as a numpy array
        perturbation_type: Type of perturbation to apply
        level: Intensity level of the perturbation
        params: Dictionary of parameters for each perturbation type
        
    Returns:
        Perturbed image
    """
    if perturbation_type == 'gaussian_noise':
        return add_gaussian_noise(image, params['std_devs'][level])
    elif perturbation_type == 'gaussian_blur':
        return apply_gaussian_blur(image, params['iterations'][level])
    elif perturbation_type == 'contrast_increase':
        return adjust_contrast(image, params['contrast_factors_inc'][level])
    elif perturbation_type == 'contrast_decrease':
        return adjust_contrast(image, params['contrast_factors_dec'][level])
    elif perturbation_type == 'brightness_increase':
        return adjust_brightness(image, params['brightness_offsets_inc'][level])
    elif perturbation_type == 'brightness_decrease':
        return adjust_brightness(image, params['brightness_offsets_dec'][level])
    elif perturbation_type == 'occlusion':
        return apply_occlusion(image, params['box_sizes'][level])
    elif perturbation_type == 'salt_pepper':
        return add_salt_pepper_noise(image, params['sp_amounts'][level])
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

def show_perturbation_examples(test_images, perturbation_type, params):
    """
    Display examples of each perturbation at different levels.
    
    Args:
        test_images: List of test image paths
        perturbation_type: Type of perturbation to apply
        params: Dictionary of parameters for each perturbation type
    """
    # Select a random image from the test set
    sample_img_path = random.choice(test_images)
    sample_img = load_image(sample_img_path)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Get the appropriate parameter list based on perturbation type
    if perturbation_type == 'gaussian_noise':
        param_list = params['std_devs']
        param_name = 'Standard Deviation'
    elif perturbation_type == 'gaussian_blur':
        param_list = params['iterations']
        param_name = 'Iterations'
    elif perturbation_type == 'contrast_increase':
        param_list = params['contrast_factors_inc']
        param_name = 'Contrast Factor'
    elif perturbation_type == 'contrast_decrease':
        param_list = params['contrast_factors_dec']
        param_name = 'Contrast Factor'
    elif perturbation_type == 'brightness_increase':
        param_list = params['brightness_offsets_inc']
        param_name = 'Brightness Offset'
    elif perturbation_type == 'brightness_decrease':
        param_list = params['brightness_offsets_dec']
        param_name = 'Brightness Offset'
    elif perturbation_type == 'occlusion':
        param_list = params['box_sizes']
        param_name = 'Box Size'
    elif perturbation_type == 'salt_pepper':
        param_list = params['sp_amounts']
        param_name = 'Noise Amount'
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    # Apply each level of perturbation and display
    for i, level in enumerate(range(10)):
        perturbed_img = apply_perturbation(sample_img, perturbation_type, level, params)
        axes[i].imshow(perturbed_img)
        axes[i].set_title(f"{param_name}: {param_list[level]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path('robustness_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save the figure
    fig.savefig(output_dir / f"{perturbation_type}_examples.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Example images saved to {output_dir / f'{perturbation_type}_examples.png'}")

def evaluate_robustness(model, device, test_images, test_masks, perturbation_type, params, img_dim=256, n_classes=3, is_point_model=False):
    """
    Evaluate the robustness of a model against a particular perturbation at different levels.
    Focuses specifically on average Dice score as the main evaluation metric.
    
    Args:
        model: Trained segmentation model
        device: Device to run evaluation on
        test_images: List of test image paths
        test_masks: List of test mask paths
        perturbation_type: Type of perturbation to apply
        params: Dictionary of parameters for perturbations
        img_dim: Image dimension for resizing
        n_classes: Number of classes in the segmentation task
        is_point_model: Whether the model is a point-based model
        
    Returns:
        Dictionary of mean dice scores for each perturbation level and the parameter list
    """
    model.eval()
    
    # Show examples of perturbations
    show_perturbation_examples(test_images, perturbation_type, params)
    
    # Get the parameter list for logging
    if perturbation_type == 'gaussian_noise':
        param_list = params['std_devs']
        param_name = 'Standard Deviation'
    elif perturbation_type == 'gaussian_blur':
        param_list = params['iterations']
        param_name = 'Iterations'
    elif perturbation_type == 'contrast_increase':
        param_list = params['contrast_factors_inc']
        param_name = 'Contrast Factor'
    elif perturbation_type == 'contrast_decrease':
        param_list = params['contrast_factors_dec']
        param_name = 'Contrast Factor'
    elif perturbation_type == 'brightness_increase':
        param_list = params['brightness_offsets_inc']
        param_name = 'Brightness Offset'
    elif perturbation_type == 'brightness_decrease':
        param_list = params['brightness_offsets_dec']
        param_name = 'Brightness Offset'
    elif perturbation_type == 'occlusion':
        param_list = params['box_sizes']
        param_name = 'Box Size'
    elif perturbation_type == 'salt_pepper':
        param_list = params['sp_amounts']
        param_name = 'Noise Amount'
    
    # Initialize dictionary to store results - focusing only on Dice scores
    dice_scores = {}
    
    # Evaluate for each perturbation level
    for level in range(10):
        logging.info(f"Evaluating {perturbation_type} with {param_name}: {param_list[level]}")
        
        # Create a perturbed dataset for this level
        perturbed_dataset = PerturbedTestDataset(
            test_images, 
            test_masks, 
            perturbation_type, 
            level, 
            params, 
            dim=img_dim,
            is_point_model=is_point_model
        )
        
        # Create a dataloader
        loader_args = dict(num_workers=4, pin_memory=True)
        perturbed_loader = DataLoader(perturbed_dataset, shuffle=False, batch_size=1, **loader_args)
        
        # Evaluate model on perturbed data
        with torch.no_grad():
            total_dice = torch.zeros(n_classes, device=device)
            batch_count = torch.zeros(n_classes, device=device, dtype=torch.float32)
            
            for batch in tqdm(perturbed_loader, desc=f"Level {level}"):
                images, masks = batch['image'], batch['mask']
                
                # Move to device
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device=device, dtype=torch.long)
                
                # Store original mask shape for later
                original_shape = masks.shape[-2:]
                
                # Handle point model input
                if is_point_model:
                    points = batch['point'].to(device=device, dtype=torch.float32)
                    outputs = model(images, points)
                else:
                    outputs = model(images)
                
                # Get predictions
                if outputs.shape[1] > 1:  # Multi-class
                    preds = outputs.argmax(dim=1)
                else:  # Binary
                    preds = (outputs > 0).squeeze(1).long()
                
                # If they don't match, resize predictions to match the original mask size
                if preds.shape[-2:] != original_shape:
                    preds = F.interpolate(
                        preds.float().unsqueeze(1),
                        size=masks.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                
                # Handle any remaining void labels (255) in mask
                masks_processed = masks.clone()
                if (masks_processed == 255).any():
                    void_pixels = (masks_processed == 255)
                    # Use most common class from predictions
                    most_common_class = torch.mode(preds[void_pixels])[0] if void_pixels.any() else 0
                    masks_processed[void_pixels] = most_common_class
                
                # Debug printing to identify shape mismatches
                if level == 0 and len(total_dice) == 0:
                    logging.info(f"Prediction shape: {preds.shape}, Mask shape: {masks_processed.shape}")
                
                # Compute Dice scores using function from metrics.py
                try:
                    dice_per_class = compute_dice_per_class(preds, masks_processed, n_classes=n_classes)
                except RuntimeError as e:
                    logging.error(f"Error computing Dice score: {e}")
                    logging.error(f"Prediction shape: {preds.shape}, Mask shape: {masks_processed.shape}")
                    # If we encounter an error, skip this batch
                    continue
                
                # Check which classes are present in this batch
                class_present = torch.zeros(n_classes, device=device, dtype=torch.bool)
                for cls in range(n_classes):
                    class_present[cls] = (masks_processed == cls).any()
                
                # Only accumulate metrics for classes that are present
                total_dice += torch.where(class_present, dice_per_class, torch.zeros_like(dice_per_class))
                batch_count += class_present.float()
            
            # Compute mean Dice score, avoiding division by zero for classes that never appeared
            batch_count = torch.maximum(batch_count, torch.ones_like(batch_count))
            mean_dice_per_class = total_dice / batch_count
            mean_dice = mean_dice_per_class.mean().item()
            
            # Store the average Dice score for this level
            dice_scores[level] = mean_dice
            
            # Log results
            logging.info(f"Level {level} ({param_name}: {param_list[level]}) - Mean Dice Score: {mean_dice:.4f}")
            # Log per-class Dice scores for more detailed analysis
            for cls in range(n_classes):
                logging.info(f"  Class {cls} - Dice: {mean_dice_per_class[cls].item():.4f}")
    
    # Return results for plotting
    return dice_scores, param_list

def plot_robustness_results(dice_scores, param_list, perturbation_type):
    """
    Plot the robustness results.
    
    Args:
        dice_scores: Dictionary of mean dice scores for each perturbation level
        param_list: List of parameter values for the x-axis
        perturbation_type: Type of perturbation applied
    """
    # Create output directory if it doesn't exist
    output_dir = Path('robustness_results')
    output_dir.mkdir(exist_ok=True)
    
    # Get values for plotting
    x_values = param_list
    y_values = [dice_scores[level] for level in range(10)]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'o-', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title based on perturbation type
    if perturbation_type == 'gaussian_noise':
        plt.xlabel('Standard Deviation')
        plt.title('Effect of Gaussian Noise on Segmentation Accuracy')
    elif perturbation_type == 'gaussian_blur':
        plt.xlabel('Number of Blur Iterations')
        plt.title('Effect of Gaussian Blur on Segmentation Accuracy')
    elif perturbation_type == 'contrast_increase':
        plt.xlabel('Contrast Factor')
        plt.title('Effect of Increased Contrast on Segmentation Accuracy')
    elif perturbation_type == 'contrast_decrease':
        plt.xlabel('Contrast Factor')
        plt.title('Effect of Decreased Contrast on Segmentation Accuracy')
    elif perturbation_type == 'brightness_increase':
        plt.xlabel('Brightness Offset')
        plt.title('Effect of Increased Brightness on Segmentation Accuracy')
    elif perturbation_type == 'brightness_decrease':
        plt.xlabel('Brightness Offset')
        plt.title('Effect of Decreased Brightness on Segmentation Accuracy')
    elif perturbation_type == 'occlusion':
        plt.xlabel('Occlusion Box Size')
        plt.title('Effect of Occlusion on Segmentation Accuracy')
    elif perturbation_type == 'salt_pepper':
        plt.xlabel('Salt and Pepper Noise Amount')
        plt.title('Effect of Salt and Pepper Noise on Segmentation Accuracy')
    
    plt.ylabel('Mean Dice Score')
    plt.ylim(0, 1.0)
    
    # Add exact values next to points
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        plt.annotate(f'{y:.3f}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / f"{perturbation_type}_robustness.png", dpi=300, bbox_inches='tight')
    print(f"Robustness results saved to {output_dir / f'{perturbation_type}_robustness.png'}")
    
    # Show the plot
    plt.show()

class PerturbedTestDataset(TestSegmentationDataset):
    """
    Dataset for testing model robustness with perturbed images.
    
    Args:
        images: List of image file paths
        masks: List of mask file paths
        perturbation_type: Type of perturbation to apply
        level: Intensity level of the perturbation
        params: Dictionary of parameters for each perturbation type
        dim: Image dimension for resizing
        is_point_model: Whether the dataset is for a point-based model
    """
    def __init__(self, images, masks, perturbation_type, level, params, dim=256, is_point_model=False):
        super().__init__(images, masks, dim=dim)
        self.perturbation_type = perturbation_type
        self.level = level
        self.params = params
        self.is_point_model = is_point_model
    
    def __getitem__(self, idx):
        # Get original image path
        img_file = self.image_files[idx]
        mask_file = self.mask_files[idx]
        
        # Load the image and mask
        mask = load_image(mask_file, is_mask=True)
        img = load_image(img_file)
        
        # Apply perturbation to the image
        perturbed_img = apply_perturbation(img, self.perturbation_type, self.level, self.params)
        
        # Apply preprocessing (using the parent class method to ensure consistency)
        from data_loading import preprocessing
        img_tensor, _, original_mask = preprocessing(perturbed_img, mask, mode='valTest', dim=self.dim)
        
        result = {
            'image': img_tensor,
            'mask': original_mask
        }
        
        # Add point information if needed for point models
        if self.is_point_model:
            # Generate a point for the mask
            from data_loading import generate_point_heatmap
            mask_np = original_mask.numpy().astype(np.uint8)
            (x, y), heatmap = generate_point_heatmap(mask_np, 10, mode='center')
            
            # Convert to tensor and add to result
            point_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)
            
            # Ensure the heatmap has the same spatial dimensions as the image
            if point_tensor.shape[-2:] != (self.dim, self.dim):
                point_tensor = F.interpolate(
                    point_tensor.unsqueeze(0),  # Add batch dimension
                    size=(self.dim, self.dim),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dimension
            
            result['point'] = point_tensor
            result['point_coords'] = torch.tensor([x, y])
        
        return result

def load_model(model_path, device, n_classes=3, model_type='unet'):
    """
    Load a trained segmentation model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
        n_classes: Number of classes in the segmentation task
        model_type: Type of model ('unet', 'point_unet', 'clip', 'autoencoder')
        
    Returns:
        Loaded model and whether it's a point model
    """
    # Create the model based on type
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=n_classes)
        is_point_model = False
    elif model_type == 'point_unet':
        model = PointUNet(n_channels=3, n_classes=n_classes)
        is_point_model = True
    elif model_type == 'clip_unet':
        model = CLIPUNet(n_classes=n_classes, bilinear=False, dropout_rate=0.0)
        is_point_model = False
    elif model_type == 'clip':
        model = CLIPSegmentationModel(n_classes=n_classes)
        is_point_model = False
    elif model_type == 'autoencoder':
        model = Autoencoder(n_channels=3, n_classes=n_classes)
        is_point_model = False
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Check which key format is used in the checkpoint
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, is_point_model

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Evaluate model robustness to various perturbations")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--img-dim', type=int, default=256, help='Image dimension')
    parser.add_argument('--model-type', type=str, default='unet', choices=['unet', 'point_unet', 'clip_unet', 'clip', 'autoencoder'], 
                        help='Model type')
    parser.add_argument('--perturbation', type=str, required=True, 
                        choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'all'], 
                        help='Perturbation type to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Define perturbation parameters
    perturbation_params = {
        'std_devs': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],  # a) Gaussian noise
        'iterations': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],      # b) Gaussian blur
        'contrast_factors_inc': [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],  # c) Contrast increase
        'contrast_factors_dec': [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],  # d) Contrast decrease
        'brightness_offsets_inc': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],  # e) Brightness increase
        'brightness_offsets_dec': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],  # f) Brightness decrease
        'box_sizes': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],  # g) Occlusion
        'sp_amounts': [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]  # h) Salt and pepper
    }
    
    # Map perturbation choice to type
    perturbation_map = {
        'a': 'gaussian_noise',
        'b': 'gaussian_blur',
        'c': 'contrast_increase',
        'd': 'contrast_decrease',
        'e': 'brightness_increase',
        'f': 'brightness_decrease',
        'g': 'occlusion',
        'h': 'salt_pepper'
    }
    
    # Select perturbations to evaluate
    if args.perturbation == 'all':
        perturbation_types = list(perturbation_map.values())
    else:
        perturbation_types = [perturbation_map[args.perturbation]]
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load the model
    logging.info(f'Loading model from {args.model_path}')
    model, is_point_model = load_model(
        args.model_path, 
        device, 
        n_classes=args.classes, 
        model_type=args.model_type
    )
    logging.info(f'Model loaded successfully')
    
    # Get test images and masks
    dir_test_img = Path('Dataset/Test/color')
    dir_test_mask = Path('Dataset/Test/label')
    
    test_images = list(dir_test_img.glob('*'))
    test_masks = list(dir_test_mask.glob('*'))
    
    # Sort and match files
    test_images, test_masks = sort_and_match_files(test_images, test_masks)
    logging.info(f'Found {len(test_images)} test images')
    
    # Evaluate robustness for each perturbation type
    for perturbation_type in perturbation_types:
        logging.info(f'Evaluating robustness for {perturbation_type}')
        
        # Evaluate
        dice_scores, param_list = evaluate_robustness(
            model, 
            device, 
            test_images, 
            test_masks, 
            perturbation_type, 
            perturbation_params, 
            img_dim=args.img_dim, 
            n_classes=args.classes,
            is_point_model=is_point_model
        )
        
        # Plot results
        plot_robustness_results(dice_scores, param_list, perturbation_type)

if __name__ == '__main__':
    main()
