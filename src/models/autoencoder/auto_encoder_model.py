import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.base.model import BaseSegmentationModel

class AutoencoderSegmentation(BaseSegmentationModel):
    """
    Autoencoder model for segmentation with dual-phase training.
    First trains for reconstruction, then fine-tunes for segmentation.
    """
    
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes)
        
        # Mode management
        self.supported_modes = ['reconstruction', 'segmentation']
        self._mode = 'reconstruction'  # Default mode
        
        # Architecture settings
        self.bilinear = bilinear
        self.use_checkpointing_flag = False
        
        # Feature dimensions for encoder
        self.enc_dims = [n_channels, 64, 128, 256, 512]
        
        # Create the encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1):
            self.encoder.append(self._make_encoder_block(self.enc_dims[i], self.enc_dims[i+1]))
        
        # Create the reconstruction decoder
        self.reconstruction_decoder = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1, 0, -1):
            self.reconstruction_decoder.append(self._make_decoder_block(self.enc_dims[i], self.enc_dims[i-1], bilinear))
        
        # Create the segmentation decoder
        self.segmentation_decoder = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1, 0, -1):
            out_channels = self.enc_dims[i-1] if i > 1 else n_classes
            self.segmentation_decoder.append(self._make_decoder_block(self.enc_dims[i], out_channels, bilinear))
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Create a convolutional block for the encoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels, bilinear=True):
        """Create a convolutional block for the decoder."""
        if bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def set_reconstruction_mode(self):
        """Set the model to reconstruction mode."""
        self.mode = 'reconstruction'
    
    def set_segmentation_mode(self, freeze_encoder_ratio=0.8):
        """
        Set the model to segmentation mode and optionally freeze encoder layers.
        
        Args:
            freeze_encoder_ratio: Ratio of encoder layers to freeze (from 0 to 1)
        """
        self.mode = 'segmentation'
        
        # Freeze a portion of the encoder layers
        params = list([p for layer in self.encoder for p in layer.parameters()])
        num_params = len(params)
        num_to_freeze = int(num_params * freeze_encoder_ratio)
        
        # Freeze the specified portion of parameters
        for i, param in enumerate(params):
            param.requires_grad = (i >= num_to_freeze)
    
    def enable_checkpointing(self, enabled=True):
        """Enable or disable gradient checkpointing to save memory."""
        self.use_checkpointing_flag = enabled
    
    def forward(self, x):
        """Forward pass based on current mode: reconstruction or segmentation."""
        features = []
        
        # Forward through encoder
        for block in self.encoder:
            if self.use_checkpointing_flag and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
            features.append(x)
        
        # Choose decoder based on mode
        if self.mode == 'reconstruction':
            # Reconstruction mode: use reconstruction decoder
            for i, block in enumerate(self.reconstruction_decoder):
                if i > 0:  # Skip connection after the first block
                    x = torch.cat([x, features[-(i+1)]], dim=1)
                
                if self.use_checkpointing_flag and self.training:
                    x = checkpoint(block, x)
                else:
                    x = block(x)
            
            return x  # Return reconstructed image
            
        else:  # segmentation mode
            # Segmentation mode: use segmentation decoder
            for i, block in enumerate(self.segmentation_decoder):
                if i > 0:  # Skip connection after the first block  
                    x = torch.cat([x, features[-(i+1)]], dim=1)
                
                if self.use_checkpointing_flag and self.training:
                    x = checkpoint(block, x)
                else: 
                    x = block(x)
            
            return x  # Return segmentation logits
    
    @property
    def mode(self):
        """Get the current mode."""
        return self._mode
    
    @mode.setter
    def mode(self, value):
        """Set the mode, with validation."""
        if value not in self.supported_modes:
            raise ValueError(f"Mode must be one of {self.supported_modes}")
        self._mode = value
