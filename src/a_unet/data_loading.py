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
    def __init__(self, images: list[str], masks: list[str], mask_suffix: str = '', dim: int = 256)
        assert len(images) == len(masks), "Mismatch between number of images and masks!"

        self.image_files = sorted(images)
        self.mask_files = sorted(masks)
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

        self.image_files = sorted(images)
        self.mask_files = sorted(masks)
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