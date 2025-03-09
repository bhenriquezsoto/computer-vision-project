#!/usr/bin/env python3
"""
Registry system for model-trainer mappings.
This allows for clean separation between models and their trainers.

Usage:
1. Import and use register_model_trainer decorator for trainer classes:
   
   @register_model_trainer('ModelClassName')
   class MyModelTrainer(BaseTrainer):
       ...

2. Get the appropriate trainer for a model:
   
   model = ModelClass(...)
   trainer_class = get_trainer_for_model(model.__class__.__name__)
   trainer = trainer_class(model, args, device)
"""
import logging
import inspect
import functools
from typing import Dict, Type, Callable, Any, Optional

# Registry to map model class names to trainer classes
_MODEL_TRAINER_REGISTRY: Dict[str, Type] = {}

def register_model_trainer(model_class_name: str) -> Callable:
    """
    Decorator to register a trainer class for a model.
    
    Args:
        model_class_name: Name of the model class this trainer supports
        
    Returns:
        Decorator function that registers the trainer
        
    Example:
        @register_model_trainer('UNet')
        class UNetTrainer(BaseTrainer):
            ...
    """
    if not model_class_name:
        raise ValueError("Model class name cannot be empty")
    if not isinstance(model_class_name, str):
        raise TypeError(f"Model class name must be a string, got {type(model_class_name)}")
    
    def decorator(trainer_class: Type) -> Type:
        # Verify trainer_class is a class
        if not inspect.isclass(trainer_class):
            raise TypeError(f"Trainer must be a class, got {type(trainer_class)}")
            
        # Verify it's a proper trainer by checking for required methods
        required_methods = ['train', 'validate', 'setup']
        missing_methods = [method for method in required_methods if not hasattr(trainer_class, method)]
        if missing_methods:
            logging.warning(f"Trainer {trainer_class.__name__} missing required methods: {missing_methods}")
        
        # Check if we're overriding an existing registration
        if model_class_name in _MODEL_TRAINER_REGISTRY:
            existing = _MODEL_TRAINER_REGISTRY[model_class_name].__name__
            logging.warning(f"Overriding existing trainer {existing} for model {model_class_name}")
        
        # Register the trainer
        _MODEL_TRAINER_REGISTRY[model_class_name] = trainer_class
        logging.info(f"Registered trainer {trainer_class.__name__} for model {model_class_name}")
        
        return trainer_class
    
    return decorator

def get_trainer_for_model(model_class_name: str) -> Type:
    """
    Get the appropriate trainer class for a model.
    
    Args:
        model_class_name: Name of the model class
        
    Returns:
        Appropriate trainer class for the model, or BaseTrainer if none is registered
        
    Example:
        trainer_class = get_trainer_for_model(model.__class__.__name__)
    """
    if not model_class_name:
        logging.error("Model class name cannot be empty")
        # Lazily import to avoid circular dependencies
        from models.base.trainer import BaseTrainer
        return BaseTrainer
    
    # Lazily import to avoid circular dependencies
    from models.base.trainer import BaseTrainer
    
    # Debug: Print registered trainers
    logging.info(f"Registered trainers: {list(_MODEL_TRAINER_REGISTRY.keys())}")
    
    if model_class_name not in _MODEL_TRAINER_REGISTRY:
        logging.warning(f"No trainer registered for model {model_class_name}. Using BaseTrainer.")
        return BaseTrainer
    
    # Get the registered trainer and verify it's valid
    trainer_class = _MODEL_TRAINER_REGISTRY[model_class_name]
    
    # Verify it's a subclass of BaseTrainer
    if not issubclass(trainer_class, BaseTrainer):
        logging.error(f"Trainer {trainer_class.__name__} for model {model_class_name} is not a subclass of BaseTrainer.")
        return BaseTrainer
    
    # Verify it has the required methods
    required_methods = ['train', 'validate', 'setup']
    missing_methods = [method for method in required_methods if not hasattr(trainer_class, method)]
    if missing_methods:
        logging.error(f"Trainer {trainer_class.__name__} missing required methods: {missing_methods}. Using BaseTrainer.")
        return BaseTrainer
    
    return trainer_class

def get_default_trainer() -> Type:
    """Get the default trainer class."""
    from models.base.trainer import BaseTrainer
    return BaseTrainer

def list_registered_trainers() -> Dict[str, str]:
    """
    List all registered model-trainer mappings.
    
    Returns:
        Dictionary mapping model class names to trainer class names
    """
    return {model: trainer.__name__ for model, trainer in _MODEL_TRAINER_REGISTRY.items()} 