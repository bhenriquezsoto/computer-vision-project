#!/usr/bin/env python3
"""
Specialized trainer for the UNet segmentation model.
"""
import logging
import torch

# Try to import wandb, but don't fail if it's not available
try:
    import wandb
except ImportError:
    logging.warning("wandb not found. Some logging features will be disabled.")
    wandb = None

from models.base.trainer import BaseTrainer
from models.base.registry import register_model_trainer

@register_model_trainer('UNet')
class UNetTrainer(BaseTrainer):
    """Specialized trainer for UNet segmentation models."""
    
    def __init__(
        self,
        model,
        args,
        device,
        dataset_dir=None,
        checkpoint_dir=None,
        project_name='U-Net',
        gradient_accumulation_steps=1,
        early_stopping_patience=10,
        early_stopping_metric='val_iou',
        early_stopping_mode='max',
        val_frequency=1,
        use_wandb=True
    ):
        """Initialize the UNet trainer."""
        super().__init__(
            model=model,
            args=args,
            device=device,
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            project_name=project_name,
            gradient_accumulation_steps=gradient_accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_mode=early_stopping_mode,
            val_frequency=val_frequency,
            use_wandb=use_wandb
        )
        
        # UNet-specific parameters
        self.bilinear = getattr(args, 'bilinear', False)
        logging.info(f"UNetTrainer initialized with bilinear upsampling: {self.bilinear}")
    
    def setup_logging(self):
        """Set up experiment logging with UNet-specific information."""
        super().setup_logging()
        
        # Add UNet-specific config to wandb
        if self.experiment is not None:
            self.experiment.config.update({
                'bilinear': self.bilinear,
            })
        
        # Additional UNet-specific logging
        logging.info(f'''UNet architecture:
            Input channels:  {self.model.n_channels}
            Output classes:  {self.model.n_classes}
            Upsampling:      {"Bilinear" if self.bilinear else "Transposed conv"}
        ''')
    
    def model_specific_setup(self):
        """UNet-specific setup steps."""
        # Add any UNet-specific setup here if needed
        pass 