#!/usr/bin/env python3
"""
Base model interface for all segmentation models.
This ensures all models have a consistent API.
"""
import torch
import torch.nn as nn
import logging

class BaseSegmentationModel(nn.Module):
    """Base class for all segmentation models."""
    
    def __init__(self, n_channels, n_classes):
        """Initialize the base segmentation model.
        
        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Mode management
        self._mode = 'default'
        self.supported_modes = ['default']
    
    @property
    def mode(self):
        """Get the current operating mode of the model."""
        return self._mode
    
    @mode.setter
    def mode(self, new_mode):
        """Set the operating mode of the model."""
        if new_mode not in self.supported_modes:
            raise ValueError(f"Unsupported mode: {new_mode}. Supported modes: {self.supported_modes}")
        self._mode = new_mode
        self._on_mode_change()
        
    def _on_mode_change(self):
        """Hook called when mode changes. Override in subclasses."""
        pass
    
    def forward(self, x, **kwargs):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            **kwargs: Additional model-specific inputs
        
        Returns:
            Output tensor of shape (B, n_classes, H, W)
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory.
        This is optional to implement in subclasses.
        """
        logging.warning("Gradient checkpointing not implemented for this model")
        pass
    
    def load_partial_weights(self, state_dict, prefix='', strict=False):
        """Load weights for a part of the model based on prefix.
        
        Args:
            state_dict: State dict containing weights
            prefix: Prefix of keys to load (e.g., 'encoder.')
            strict: Whether to strictly enforce that the keys match
            
        Returns:
            tuple: (missing_keys, unexpected_keys)
        """
        # Filter weights by prefix
        filtered_dict = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not filtered_dict:
            raise ValueError(f"No weights found with prefix '{prefix}'")
        
        # Strip prefix if needed
        if prefix:
            filtered_dict = {k[len(prefix):]: v for k, v in filtered_dict.items()}
        
        # Get the target module based on prefix
        if prefix and '.' in prefix:
            # Handle nested modules (e.g., 'encoder.layer1')
            target = self
            parts = prefix.rstrip('.').split('.')
            for part in parts:
                if not part:
                    continue
                if not hasattr(target, part):
                    raise AttributeError(f"Model has no attribute '{part}' in path '{prefix}'")
                target = getattr(target, part)
            return target.load_state_dict(filtered_dict, strict=strict)
        else:
            # Load directly into model (or just one level, like 'encoder')
            target = self if not prefix else getattr(self, prefix.rstrip('.'))
            return target.load_state_dict(filtered_dict, strict=strict)
    
    @property
    def trainer_class(self):
        """Get the appropriate trainer class for this model.
        
        Note: This method is deprecated. Use registry.get_trainer_for_model() instead.
        It's kept for backward compatibility.
        """
        # Import here to avoid circular imports
        from models.base.registry import get_trainer_for_model
        
        # This gets the appropriate trainer from the registry based on class name
        return get_trainer_for_model(self.__class__.__name__) 