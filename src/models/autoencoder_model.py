import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    Autoencoder model for image segmentation that uses a two-phase training approach:
    1. Reconstruction phase: Train the encoder-decoder to reconstruct the input image
    2. Segmentation phase: Reuse the encoder and train a new decoder for segmentation
    
    This model can be configured to output the appropriate number of classes
    and will handle the training phase internally.
    """
    def __init__(self, n_channels=3, n_classes=3):
        super(Autoencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.training_phase = "segmentation"  # Default to segmentation phase
        
        # Register normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 1024),
        )
        
        # Reconstruction decoder (outputs RGB image)
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(1024, 64 * 32 * 32),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, n_channels, kernel_size=3, stride=1, padding=1),
        )
        
        # Segmentation decoder (outputs segmentation classes)
        self.segmentation_decoder = nn.Sequential(
            nn.Linear(1024, 64 * 32 * 32),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, n_classes, kernel_size=3, stride=1, padding=1),
        )
        
    def set_phase(self, phase):
        """Set the training phase ('reconstruction' or 'segmentation')"""
        if phase not in ["reconstruction", "segmentation"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'reconstruction' or 'segmentation'")
        self.training_phase = phase
        return self
        
    def forward(self, x):
        """
        Forward pass through the model
        Returns reconstruction or segmentation output based on current phase
        """
        x = self.encoder(x)
        
        if self.training_phase == "reconstruction":
            return self.reconstruction_decoder(x)
        else:  # segmentation phase
            return self.segmentation_decoder(x)
        
    def load_pretrained_encoder(self, state_dict):
        """
        Load pretrained weights for the encoder from a reconstruction-trained model
        
        Args:
            state_dict: State dict from a pretrained model
        """
        # Extract only the encoder weights from the state dict
        encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
        
        # Load the encoder weights
        missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading encoder: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading encoder: {unexpected_keys}")
            
        return self
