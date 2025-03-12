import logging
import numpy as np
import torch
from PIL import Image
from multiprocessing import Pool
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F

    
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
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', dim: int = 256, mode: str = 'train'):
        assert len(images) == len(masks), "Mismatch between number of images and masks!"

        # Store the files directly, assuming they are already matched
        self.image_files = images
        self.mask_files = masks
        
        self.mask_suffix = mask_suffix
        self.dim = dim
        self.mode = mode
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
        img, mask, _ = preprocessing(img, mask, mode=self.mode, dim=self.dim)
        
        return {
            'image': img,
            'mask': mask
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


class PointSegmentationDataset(Dataset):
    """Point prompt segmentation dataset that generates point prompts for segmentation.
    For each image-mask pair, it:
    1. Randomly selects either cat (1) or dog (2) class
    2. Samples a random point within that class region
    3. Creates a Gaussian heatmap centered at that point
    4. Returns the image, point heatmap, and corresponding binary mask
    """
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', dim: int = 256, sigma: float = 10.0):
        assert len(images) == len(masks), "Mismatch between number of images and masks!"

        # Store the files directly, assuming they are already matched
        self.image_files = images
        self.mask_files = masks
        self.mask_suffix = mask_suffix
        self.dim = dim
        self.sigma = sigma  # Standard deviation for Gaussian heatmap
        
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

    def generate_gaussian_heatmap(self, center_y: int, center_x: int, height: int, width: int) -> np.ndarray:
        """Generate a Gaussian heatmap centered at (center_y, center_x)."""
        y = np.arange(0, height, 1, float)
        x = np.arange(0, width, 1, float)
        y, x = np.meshgrid(y, x)
        
        # Generate 2D gaussian
        heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * self.sigma ** 2))
        return heatmap

    def sample_point_from_mask(self, mask: np.ndarray, target_class: int) -> tuple[int, int]:
        """Sample a random point from the region of target_class in the mask."""
        # Get coordinates where mask equals target class
        y_coords, x_coords = np.where(mask == target_class)
        
        if len(y_coords) == 0:
            raise ValueError(f"No pixels found for class {target_class}")
            
        # Randomly select one point
        idx = np.random.randint(0, len(y_coords))
        return y_coords[idx], x_coords[idx]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.mask_files[idx]
        
        # Load the image and mask
        mask = load_image(mask_file, is_mask=True)
        img = load_image(img_file)
        
        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {img_file}, {mask_file} should be the same size, but are {img.shape[:2]} and {mask.shape[:2]}'
            
        # Randomly select either cat (1) or dog (2)
        available_classes = []
        if 1 in mask:  # Check if cat exists
            available_classes.append(1)
        if 2 in mask:  # Check if dog exists
            available_classes.append(2)
            
        if not available_classes:
            raise ValueError(f"No cat or dog found in mask {mask_file}")
            
        target_class = np.random.choice(available_classes)
        
        # Sample a random point from the selected class region
        point_y, point_x = self.sample_point_from_mask(mask, target_class)
        
        # Create point heatmap
        heatmap = self.generate_gaussian_heatmap(point_y, point_x, mask.shape[0], mask.shape[1])
        
        # Create binary mask for the target class
        binary_mask = (mask == target_class).astype(np.float32)
        
        # Apply preprocessing to image, heatmap, and binary mask
        img, mask, original_mask = preprocessing(img, binary_mask, mode='train', dim=self.dim)
        
        # Convert heatmap to tensor and resize to match preprocessed image
        heatmap_tensor = torch.from_numpy(heatmap).float()
        heatmap_tensor = heatmap_tensor.unsqueeze(0)  # Add channel dimension
        heatmap_tensor = F.interpolate(
            heatmap_tensor.unsqueeze(0),  # Add batch dimension
            size=(self.dim, self.dim),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        return {
            'image': img,  # Shape: (3, H, W)
            'point_heatmap': heatmap_tensor,  # Shape: (1, H, W)
            'mask': mask  # Shape: (H, W)
        }

