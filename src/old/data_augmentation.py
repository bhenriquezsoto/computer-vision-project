import os
from skimage.io import imread
from torch.utils.data import Dataset
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_filenames)
    
    def get_label_filename(self, idx):
        return self.image_filenames[idx].replace(".jpg", ".png")
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.get_label_filename(idx))

        image = imread(image_path)
        mask = imread(mask_path)

        # Convert RGBA to RGB if necessary
        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Convert masks from (H, W, 3) to (H, W) based on the color of the pixel
        COLOR_MAP = {
            (0, 0, 0): 0,  # Background
            (128, 0, 0): 1,  # Red (Cat)
            (0, 128, 0): 2  # Green (Dog)
        }

        label_mask = np.full(mask.shape[:2], 3, dtype=np.uint8)

        for rgb, label in COLOR_MAP.items():
            mask_pixels = (mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1]) & (mask[:, :, 2] == rgb[2])
            label_mask[mask_pixels] = label

        return image, label_mask