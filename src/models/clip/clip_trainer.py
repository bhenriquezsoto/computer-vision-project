#!/usr/bin/env python3
"""
Specialized trainer for CLIP-based segmentation models.
This extends the BaseTrainer with CLIP-specific functionality.
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

@register_model_trainer('CLIPSegmentationModel')
class CLIPTrainer(BaseTrainer):
    """Trainer specialized for CLIP-based segmentation models."""
    
    def __init__(
        self,
        model,
        args,
        device,
        dataset_dir=None,
        checkpoint_dir=None,
        project_name='CLIP-Segmentation',
        gradient_accumulation_steps=1,
        early_stopping_patience=10,
        early_stopping_metric='val_iou',
        early_stopping_mode='max',
        val_frequency=1,
        use_wandb=True
    ):
        """Initialize the CLIP trainer."""
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
        
        # CLIP-specific parameters
        self.train_clip_backbone = getattr(args, 'train_clip_backbone', False)
        self.clip_pretrained = getattr(model, 'pretrained', True)
        logging.info(f"CLIPTrainer initialized with pretrained backbone: {self.clip_pretrained}")
    
    def setup_training(self):
        """Set up training components with CLIP-specific settings."""
        super().setup_training()
        
        # Special handling for fine-tuning CLIP
        # By default, we freeze the CLIP backbone and only train the segmentation head
        if hasattr(self.model, 'clip_model') and not self.train_clip_backbone:
            for param in self.model.clip_model.parameters():
                param.requires_grad = False
            logging.info("CLIP backbone frozen for training")
    
    def setup_logging(self):
        """Set up experiment logging with CLIP-specific information."""
        super().setup_logging()
        
        # Add CLIP-specific config to wandb
        if self.experiment is not None:
            self.experiment.config.update({
                'clip_pretrained': self.clip_pretrained,
                'train_clip_backbone': self.train_clip_backbone
            })
        
        # Additional CLIP-specific logging
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f'''CLIP Segmentation model:
            Output classes:  {self.model.n_classes}
            Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})
            Backbone frozen: {not self.train_clip_backbone}
        ''') 