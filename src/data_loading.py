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
from scipy.ndimage import gaussian_filter

    
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

def calculate_class_weights(mask_files, n_classes):
    """
    Calculate class weights based on class frequencies in the dataset.
    Weights are inversely proportional to class frequencies with moderate balancing.
    
    Args:
        mask_files (list): List of mask file paths
        n_classes (int): Number of classes
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = torch.zeros(n_classes)
    total_pixels = 0
    
    for mask_file in mask_files:
        mask = load_image(mask_file, is_mask=True)
        # Count pixels for each class (excluding void pixels)
        for c in range(n_classes):
            class_counts[c] += (mask == c).sum()
        total_pixels += (mask != 255).sum()
    
    # Calculate weights (inverse of frequency)
    class_weights = total_pixels / (n_classes * class_counts + 1e-10)
    
    # Apply moderate balancing (less extreme than power 1.5)
    class_weights = class_weights ** 0.75
    
    # Ensure no class gets too little attention by setting a minimum weight threshold
    min_weight = class_weights.max() * 0.2
    class_weights = torch.maximum(class_weights, torch.tensor(min_weight))
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    return class_weights
    
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
            # A.ElasticTransform(alpha=1, sigma=50, p=0.1),  # Elastic distortion
            # A.GridDistortion(p=0.3),  # Slight grid warping
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Color jitter
            # A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Random blur
            # A.GaussNoise(var_limit=(10, 50), p=0.2),  # Random noise
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

def create_point_heatmap(point, shape, sigma=3.0):
    """
    Create a Gaussian heatmap centered at the provided point.
    
    Args:
        point (tuple): (y, x) coordinates of the center point
        shape (tuple): (height, width) of the output heatmap
        sigma (float): Standard deviation of the Gaussian kernel
        
    Returns:
        np.ndarray: Gaussian heatmap
    """
    y, x = point
    heatmap = np.zeros(shape, dtype=np.float32)
    
    # Place a single 1 at the point location
    if 0 <= y < shape[0] and 0 <= x < shape[1]:
        heatmap[y, x] = 1.0
    
    # Apply Gaussian filter to spread the point
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

class UnifiedPointSegmentationDataset(Dataset):
    """
    Unified dataset for point-based segmentation, supporting both training and testing modes.
    
    Args:
        images (list[str]): List of image filenames
        masks (list[str]): List of mask filenames
        mode (str): 'train' or 'test' to determine point sampling strategy
        dim (int): Target image dimension
        sigma (float): Standard deviation for Gaussian point heatmap
    """
    def __init__(self, images: list[str], masks: list[str], mode: str = 'train', dim: int = 256, sigma: float = 3.0):
        assert len(images) == len(masks), "Mismatch between number of images and masks!"
        assert mode in ['train', 'test'], f"Mode must be 'train' or 'test', got {mode}"
        
        self.image_files = images
        self.mask_files = masks
        self.mode = mode
        self.dim = dim
        self.sigma = sigma
        
        logging.info(f'Creating {mode} point-based segmentation dataset with {len(self.image_files)} examples')
        
        # Scan for unique mask values
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
        
        # Load image and mask
        mask = load_image(mask_file, is_mask=True)
        img = load_image(img_file)
        
        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {img_file}, {mask_file} should be the same size, but are {img.shape[:2]} and {mask.shape[:2]}'
        
        # Apply preprocessing transformations based on mode
        preprocess_mode = 'train' if self.mode == 'train' else 'valTest'
        img_tensor, mask_tensor, original_mask = preprocessing(img, mask, mode=preprocess_mode, dim=self.dim)
        
        # Convert tensors back to numpy for point sampling
        # We need to work with the processed mask to ensure alignment
        processed_mask_np = mask_tensor.numpy()
        
        # Find available classes in this image (excluding void pixels)
        available_classes = np.unique(processed_mask_np)
        available_classes = available_classes[available_classes < 255]  # Remove void class
        
        # If no classes are available (rare case), default to background
        if len(available_classes) == 0:
            available_classes = np.array([0])
        
        # Randomly select a class from available classes (same for both modes)
        target_class = np.random.choice(available_classes)
        
        # Find coordinates for the specified class
        y_coords, x_coords = np.where(processed_mask_np == target_class)
        
        # Create binary mask for the target class
        target_mask = (processed_mask_np == target_class).astype(np.float32)
        
        # If no pixels of this class are found (which shouldn't happen now), generate point near center
        if len(y_coords) == 0:
            # This should be rare since we're selecting from available classes
            y = processed_mask_np.shape[0] // 2
            x = processed_mask_np.shape[1] // 2
        else:
            if self.mode == 'train':
                # For training: random point from the class
                point_idx = np.random.randint(0, len(y_coords))
                y, x = y_coords[point_idx], x_coords[point_idx]
            else:
                # For testing: use centroid of the class for more stable evaluation
                y = int(np.mean(y_coords))
                x = int(np.mean(x_coords))
        
        # Create point heatmap
        point_heatmap = create_point_heatmap((y, x), processed_mask_np.shape, sigma=self.sigma)
        
        # Convert heatmap to tensor
        point_heatmap_tensor = torch.from_numpy(point_heatmap).unsqueeze(0)  # Add channel dimension [1, H, W]
        
        # Convert target mask to tensor
        target_mask_tensor = torch.from_numpy(target_mask).long()
        
        result = {
            'image': img_tensor,  # [3, H, W]
            'point': point_heatmap_tensor,  # [1, H, W]
            'mask': target_mask_tensor,  # [H, W]
            'class': target_class  # Class ID for reference
        }
        
        # Add image index for test mode
        if self.mode == 'test':
            result['image_idx'] = idx
            
        return result

class PointSegmentationDataset(UnifiedPointSegmentationDataset):
    """
    Dataset for point-based segmentation that samples points on objects and provides
    (image, point_heatmap, mask) triplets for training.
    
    Args:
        images (list[str]): List of image filenames
        masks (list[str]): List of mask filenames
        dim (int): Target image dimension
        sigma (float): Standard deviation for Gaussian point heatmap
    """
    def __init__(self, images: list[str], masks: list[str], dim: int = 256, sigma: float = 3.0):
        super().__init__(images, masks, mode='train', dim=dim, sigma=sigma)


class TestPointSegmentationDataset(UnifiedPointSegmentationDataset):
    """
    Dataset for testing point-based segmentation models.
    
    Args:
        images (list[str]): List of image filenames
        masks (list[str]): List of mask filenames
        dim (int): Target image dimension
        sigma (float): Standard deviation for Gaussian point heatmap
    """
    def __init__(self, images: list[str], masks: list[str], dim: int = 256, sigma: float = 3.0):
        super().__init__(images, masks, mode='test', dim=dim, sigma=sigma)