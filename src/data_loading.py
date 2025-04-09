import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import random
import torch.nn.functional as F
from scipy import ndimage

    
def load_image(filename, is_mask=False):
    """Load an image while preserving grayscale images as single-channel.
    
    Supports:
    - `.npy`: NumPy array
    - `.pt` / `.pth`: PyTorch tensor
    - Standard images (`.png`, `.jpg`, etc.)

    Returns:
        np.ndarray: Image or mask as NumPy array.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Error: File {filename} not found!")
    
    ext = splitext(filename)[1]
        
    if ext == '.npy':
        img = np.asarray(Image.fromarray(np.load(filename)))
    elif ext in ['.pt', '.pth']:
        img = np.asarray(Image.fromarray(torch.load(filename).numpy()))
    else:
        img = np.asarray(Image.open(filename))
    
    # Remove the alpha channel if it exists
    if not is_mask and img.ndim == 3 and img.shape[2] == 4:
        img = img[:,:,:3]
        
    return img

    
def unique_mask_values(mask_file):
    """Load mask file and return unique values."""
    mask = np.asarray(load_image(mask_file, is_mask=True))
    
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def fill_void_labels_with_neighbor_info(mask, void_label=255):
    """
    Fill void label pixels with the most common neighboring label.
    
    Args:
        mask (np.ndarray): Input mask with void label pixels
        void_label (int): Value representing void label (default: 255)
    
    Returns:
        np.ndarray: Mask with void label pixels filled based on neighborhood analysis
    """
    # Create a copy to avoid modifying the original
    filled_mask = mask.copy()
    
    # Find void label pixels
    void_pixels = (mask == void_label)
    
    # If no void pixels, return the original mask
    if not np.any(void_pixels):
        return filled_mask
    
    # For each class, calculate a distance transform to find the nearest pixel of that class
    # Start with a high initial distance
    min_distances = np.full_like(mask, float('inf'), dtype=float)
    best_class = np.zeros_like(mask)
    
    # Get unique classes (excluding void label)
    classes = np.unique(mask)
    classes = classes[classes != void_label]
    
    for cls in classes:
        # Create a binary mask for this class
        class_mask = (mask == cls)
        
        # Calculate distance transform - how far each pixel is from this class
        dist = ndimage.distance_transform_edt(~class_mask)
        
        # Update min_distances and best_class where this class is closer
        closer = dist < min_distances
        min_distances[closer] = dist[closer]
        best_class[closer] = cls
    
    # Replace void pixels with their nearest class
    filled_mask[void_pixels] = best_class[void_pixels]
    
    return filled_mask

def preprocessing(img: np.ndarray, mask: np.ndarray, mode: str = 'train', dim: int = 256):
    """Preprocess the image and mask for training.
    mode 'train' is for training, applying data augmentation and resizing.
    mode 'valTest' is for validation/testing, applying normalization only.
    
    Args:
        img (np.ndarray): Image as NumPy array.
        mask (np.ndarray): Mask as NumPy array.
        dim (int): Target image dimension.
        mode (str): One of 'train', 'valTest'.
        
    Returns:
        Tuple[torch.Tensor]: Processed image and mask as PyTorch tensors.
    """
    
    assert mode in ['train', 'valTest'], f'Invalid mode: {mode}'
    assert dim > 0, f'Invalid image dimension: {dim}'
    
    # Process void labels (255) based on neighborhood before augmentation
    if mask is not None:
        mask = fill_void_labels_with_neighbor_info(mask, void_label=255)
    
    # Define common transformations for standard resizing and normalization    
    resizing = A.Compose([
        A.LongestMaxSize(max_size=dim, interpolation=0),
        A.PadIfNeeded(min_height=dim, min_width=dim, border_mode=0)
    ])
    normalisation = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    if mode == 'valTest':
        augmentation = A.Compose([
            resizing,
            normalisation,
            ToTensorV2()
        ])
    else:
        # Define transformations for augmentation
        augmentation = A.Compose([
            resizing,
            
            #### ADD AUGMENTATION HERE ####
            
            # A.RandomCrop(img_dim, img_dim),  # Crop to fixed size
            A.HorizontalFlip(p=0.5),  # Flip images & masks with 50% probability
            A.Rotate(limit=20, p=0.5),  # Random rotation (-20° to 20°)
            A.ElasticTransform(p=0.2),  # Elastic distortion
            A.GridDistortion(p=0.2),  # Slight grid warping
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4), # color jitter
            A.GaussNoise(var_limit=(10, 50), p=0.3), # Random Gaussian noise
            # A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Random blur
            # A.CoarseDropout(max_holes=2, max_height=50, max_width=50, p=0.3),  # Cutout occlusion
            
            ### END AUGMENTATION ###
            
            normalisation, 
            ToTensorV2()  # Convert to PyTorch tensor
        ])
        
    original_mask = torch.tensor(mask, dtype=torch.long)
    augmented = augmentation(image=img, mask=mask)
    return augmented['image'], augmented['mask'], original_mask

class SegmentationDataset(Dataset):
    """General segmentation dataset for different datasets, supporting transforms and scaling.

    Args:
        images_dir (list[str]): List of image filenames.
        mask_dir (list[str]): List of mask filenames.
        mask_suffix (str): Suffix used in mask filenames.
        transform (albumentations.Compose, optional): Data augmentation pipeline. Defaults to None. If none, defaultly resize the image to 256x256 and normalize it.
        scale (float, optional): Scaling factor for resizing. Defaults to None.
    """
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', dim: int = 256):
        assert len(images) == len(masks), "Mismatch between number of images and masks!"

        # Store the files directly, assuming they are already matched
        self.image_files = images
        self.mask_files = masks
        
        self.mask_suffix = mask_suffix
        self.dim = dim
        
        logging.info(f'Creating dataset with {len(self.image_files)} examples')
        logging.info('Scanning mask files to determine unique values')

        # Use `masks` list directly instead of searching a directory
        with Pool() as p:
            unique = list(tqdm(
                p.imap(unique_mask_values, self.mask_files),
                total=len(self.mask_files)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.mask_files[idx]
        
        # Load the image and mask in PIL format 
        mask = load_image(mask_file, is_mask=True)
        img = load_image(img_file)
        
        
        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {img_file}, {mask_file} should be the same size, but are {img.shape[:2]} and {mask.shape[:2]}'
            
        # Apply the transformations for data augmentation and/or preprocessing
        img, mask, _ = preprocessing(img, mask, mode='train', dim=self.dim)
        
        return {
            'image': img,
            'mask': mask
        }

class TestSegmentationDataset(Dataset):
    """General segmentation dataset for test and validation datasets, keeping the original mask.

    Args:
        images_dir (list[str]): List of image filenames.
        mask_dir (list[str]): List of mask filenames.
        mask_suffix (str): Suffix used in mask filenames.
        dim (int): Target image dimension.
        
    Returns:
        Tuple[torch.Tensor]: Processed image and mask as PyTorch tensors.
    """
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', dim: int = 256):
        assert len(images) == len(masks), "Mismatch between number of images and masks!"

        # Store the files directly, assuming they are already matched
        self.image_files = images
        self.mask_files = masks
        
        self.mask_suffix = mask_suffix
        self.dim = dim
        
        logging.info(f'Creating dataset with {len(self.image_files)} examples')
        logging.info('Scanning mask files to determine unique values')

        # Use `masks` list directly instead of searching a directory
        with Pool() as p:
            unique = list(tqdm(
                p.imap(unique_mask_values, self.mask_files),
                total=len(self.mask_files)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.mask_files[idx]
        
        # Load the image and mask in PIL format 
        mask = load_image(mask_file, is_mask=True)
        img = load_image(img_file)
        
        
        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {img_file} should be the same size, but are {img.shape[:2]} and {mask.shape[:2]}'
            
        # Apply the transformations for data augmentation and/or preprocessing
        img, _, original_mask = preprocessing(img, mask, mode='valTest', dim=self.dim)
        
        return {
            'image': img,
            'mask': original_mask
        }

def sort_and_match_files(images, masks):
    """
    Sort and match image and mask files by their base names.
    
    Args:
        images: List of image file paths
        masks: List of mask file paths
        
    Returns:
        matched_images: List of matched and sorted image file paths
        matched_masks: List of matched and sorted mask file paths
    """
    print("Sorting and matching files...")
    # Create dictionaries with base names as keys
    image_dict = {Path(img).stem: img for img in images}
    mask_dict = {Path(mask).stem: mask for mask in masks}
    
    # Find common base names
    common_names = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
    
    if len(common_names) < min(len(images), len(masks)):
        logging.warning(f'Only {len(common_names)} out of {min(len(images), len(masks))} image-mask pairs matched by name!')
        logging.warning(f'Example image names: {list(image_dict.keys())[:5]}')
        logging.warning(f'Example mask names: {list(mask_dict.keys())[:5]}')
    
    # Create paired lists of matched files
    matched_images = [image_dict[name] for name in common_names]
    matched_masks = [mask_dict[name] for name in common_names]
    
    return matched_images, matched_masks

# Add a new function to generate a point heatmap
def generate_point_heatmap(mask, sigma=10, mode='random', class_weights=None):
    """
    Generate a point heatmap for a mask with improved class sampling.
    
    Args:
        mask (np.ndarray): Segmentation mask
        sigma (int): Standard deviation for Gaussian kernel
        mode (str): Point selection mode ('random', 'center', 'weighted')
        class_weights (list, optional): Weights for class sampling probabilities
    
    Returns:
        tuple: (x, y) coordinates of the point and heatmap
    """
    # Get unique class values excluding void label (255)
    unique_values = np.unique(mask)
    valid_classes = [val for val in unique_values if val != 255]
    
    # Check specifically for cats (class 1)
    has_cat = 1 in valid_classes
    
    # Give higher priority to cats when present
    if has_cat and random.random() < 0.7:  # 70% chance to select cat when present
        selected_class = 1
    elif mode == 'weighted' and class_weights is not None:
        # Use class weights to determine sampling probabilities
        # Weight sampling toward underrepresented classes
        class_probs = {}
        for cls in valid_classes:
            # Use inverse frequency as weight (modified by provided class_weights)
            if cls < len(class_weights):  # Ensure class has a weight defined
                cls_weight = class_weights[cls]
                class_probs[cls] = cls_weight
        
        # Normalize probabilities
        total = sum(class_probs.values())
        if total > 0:
            for cls in class_probs:
                class_probs[cls] /= total
                
            # Select class based on weighted probability
            classes = list(class_probs.keys())
            probs = [class_probs[cls] for cls in classes]
            selected_class = np.random.choice(classes, p=probs)
        else:
            # Fallback to random selection
            selected_class = random.choice(valid_classes)
    else:
        # Standard random selection
        selected_class = random.choice(valid_classes)
    
    # Get coordinates of pixels belonging to the selected class
    y_indices, x_indices = np.where(mask == selected_class)
    
    if len(y_indices) == 0:
        # Fallback: use the center of the image
        height, width = mask.shape
        x, y = width // 2, height // 2
    else:
        if mode == 'center':
            # Use the center of mass of the selected class
            y = int(np.mean(y_indices))
            x = int(np.mean(x_indices))
        else:  # 'random' or 'weighted'
            # Randomly select a point from the selected class
            idx = random.randint(0, len(y_indices) - 1)
            y, x = y_indices[idx], x_indices[idx]
    
    # Create a heatmap using a Gaussian kernel
    height, width = mask.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[y, x] = 1
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Normalize the heatmap
    heatmap = heatmap / heatmap.max()
    
    return (x, y), heatmap

class PointSegmentationDataset(SegmentationDataset):
    """
    Segmentation dataset that includes point prompts.
    
    Args:
        images: List of image file paths
        masks: List of mask file paths
        mask_suffix: Suffix for mask files
        dim: Image dimension for resizing
        sigma: Standard deviation for Gaussian kernel
        point_mode: Point selection mode ('random', 'center', 'weighted')
        class_weights: Weights for class sampling probabilities
    """
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', 
                 dim: int = 256, sigma: int = 10, point_mode: str = 'weighted',
                 class_weights=None):
        super().__init__(images, masks, mask_suffix, dim)
        self.sigma = sigma
        self.point_mode = point_mode
        
        # Set default weights that prioritize cat class (class 1)
        if class_weights is None:
            self.class_weights = [0.5, 2.0, 0.8]  # Background, Cat, Dog - more emphasis on cats
        else:
            self.class_weights = class_weights
            
        logging.info(f"PointSegmentationDataset initialized with class weights: {self.class_weights}")
        
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        img, mask = result['image'], result['mask']
        
        # Convert mask back to numpy for point generation
        mask_np = mask.numpy().astype(np.uint8)
        
        # Generate a point and heatmap with class weights, using weighted mode
        (x, y), heatmap = generate_point_heatmap(
            mask_np, 
            self.sigma, 
            mode=self.point_mode, 
            class_weights=self.class_weights
        )
        
        # Convert heatmap to tensor
        point_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)
        
        # Add point and heatmap to result
        result['point'] = point_tensor
        result['point_coords'] = torch.tensor([x, y])
        
        return result

class TestPointSegmentationDataset(TestSegmentationDataset):
    """
    Test dataset for point-based segmentation.
    
    Args:
        images: List of image file paths
        masks: List of mask file paths
        mask_suffix: Suffix for mask files
        dim: Image dimension for resizing
        sigma: Standard deviation for Gaussian kernel
        point_mode: Point selection mode ('random', 'center', 'weighted')
        class_weights: Weights for class sampling probabilities
    """
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', 
                 dim: int = 256, sigma: int = 10, point_mode: str = 'weighted',
                 class_weights=None):
        super().__init__(images, masks, mask_suffix, dim)
        self.sigma = sigma
        self.point_mode = point_mode
        
        # Set default weights that prioritize cat class (class 1)
        if class_weights is None:
            self.class_weights = [0.5, 2.0, 0.8]  # Background, Cat, Dog - more emphasis on cats
        else:
            self.class_weights = class_weights
            
        logging.info(f"TestPointSegmentationDataset initialized with class weights: {self.class_weights}")
        
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        img, mask = result['image'], result['mask']
        
        # Convert mask to numpy for point generation
        mask_np = mask.numpy().astype(np.uint8)
        
        # Generate a point and heatmap with class weights
        (x, y), heatmap = generate_point_heatmap(
            mask_np, 
            self.sigma, 
            mode=self.point_mode, 
            class_weights=self.class_weights
        )
        
        # Resize heatmap to match the image dimensions
        # This is critical because the image might have been resized in preprocessing
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)
        
        # Ensure the heatmap has the same spatial dimensions as the image
        # The image should be (C, H, W) where H and W are both self.dim
        if heatmap_tensor.shape[-2:] != (self.dim, self.dim):
            heatmap_tensor = F.interpolate(
                heatmap_tensor.unsqueeze(0),  # Add batch dimension
                size=(self.dim, self.dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
        
        # Add point and heatmap to result
        result['point'] = heatmap_tensor
        result['point_coords'] = torch.tensor([x, y])
        
        return result