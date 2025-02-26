import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb
from evaluate import evaluate, compute_dice_per_class, compute_iou_per_class, compute_pixel_accuracy, dice_loss
from unet_model import UNet
from data_loading import SegmentationDataset

dir_img = Path('Dataset/TrainVal/color')
dir_mask = Path('Dataset/TrainVal/label')
dir_test_img = Path('Dataset/Test/color')
dir_test_mask = Path('Dataset/Test/label')
dir_checkpoint = Path('src/a_unet/checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        optimizer: str = 'adam',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = False,
        img_scale: float = 1,
        amp: bool = False,
        gradient_clipping: float = 1.0
):
    # 1. Create dataset

    # 1.1 Define transformations for data preprocessing and augmentation
    transform = A.Compose([
        ######### TODO: Maybe take out this fist padding ##########
        # A.PadIfNeeded(min_height=300, min_width=300, border_mode=0, value=(0, 0, 0)),  # Pad small images to 300x300
        ######################################################
        A.LongestMaxSize(max_size=256, interpolation=0),  # Resize longest side to 256 (if necessary)
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),  # Pad remaining images to 256x256
        A.RandomCrop(256, 256),  # Crop to fixed size
        A.HorizontalFlip(p=0.5),  # Flip images & masks with 50% probability
        A.Rotate(limit=20, p=0.5),  # Random rotation (-20° to 20°)
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # Elastic distortion
        # A.GridDistortion(p=0.3),  # Slight grid warping
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Color jitter
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Random blur
        # A.GaussNoise(var_limit=(10, 50), p=0.2),  # Random noise
        # A.CoarseDropout(max_holes=2, max_height=50, max_width=50, p=0.3),  # Cutout occlusion
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard normalization
        ToTensorV2()  # Convert to PyTorch tensor
    ])

    try:
        dataset = SegmentationDataset(dir_img, dir_mask, transform=transform)
    except (AssertionError, RuntimeError, IndexError):
        print("SegmentationDataset failed on training set, check data_loading.py")

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    print("Training set dimensions: ", len(train_set))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Weight decay:    {weight_decay}
        Optimizer:       {optimizer}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.999)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.999)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # or ReduceLROnPlateau(optimizer, mode='max', patience=5)
    grad_scaler = GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(ignore_index=255) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        best_val_dice = 0
        
        total_dice = torch.zeros(model.n_classes, device=device)  # Store per-class Dice
        total_iou = torch.zeros(model.n_classes, device=device)   # Store per-class IoU
        total_acc = 0  
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', leave=True) as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has {model.n_channels} input channels, but got {images.shape[1]}.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(masks_pred.squeeze(1), true_masks.float(), n_classes=model.n_classes)
                    else:
                        true_masks_processed = true_masks.clone()
                        true_masks_processed[true_masks_processed == 255] = 0  # Ignore void label
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(masks_pred, true_masks_processed, n_classes=model.n_classes)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                # Compute per-class IoU, Dice Score, and Pixel Accuracy
                dice_scores = compute_dice_per_class(masks_pred.argmax(dim=1), true_masks_processed, n_classes=model.n_classes)
                iou_scores = compute_iou_per_class(masks_pred.argmax(dim=1), true_masks_processed, n_classes=model.n_classes)
                pixel_acc = compute_pixel_accuracy(masks_pred.argmax(dim=1), true_masks_processed)

                total_dice += dice_scores
                total_iou += iou_scores
                total_acc += pixel_acc

                # Log training metrics
                experiment.log({
                    'train loss': loss.item(),
                    'train Dice (avg)': dice_scores.mean().item(),
                    'train IoU (avg)': iou_scores.mean().item(),
                    'train Pixel Accuracy': pixel_acc,
                    **{f'train Dice class {i}': dice_scores[i].item() for i in range(model.n_classes)},
                    **{f'train IoU class {i}': iou_scores[i].item() for i in range(model.n_classes)},
                    'step': global_step,
                    'epoch': epoch
                })

                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice_scores.mean().item():.4f}", iou=f"{iou_scores.mean().item():.4f}")
                
        # Compute average training metrics
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        epoch_loss /= len(train_loader)

        logging.info(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}, Dice Score: {avg_dice.mean().item():.4f}, IoU: {avg_iou.mean().item():.4f}, Pixel Acc: {avg_acc:.4f}")

        # Perform validation at the end of each epoch
        val_dice, val_iou, val_acc, val_dice_per_class, val_iou_per_class = evaluate(model, val_loader, device, amp, n_classes=model.n_classes)
        
        optimizer.step()
        
        # Update scheduler (if using ReduceLROnPlateau)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_iou)
        else:
            scheduler.step()

        # Log validation metrics
        experiment.log({
            'validation Dice (avg)': val_dice,
            'validation IoU (avg)': val_iou,
            'validation Pixel Accuracy': val_acc,
            **{f'validation Dice class {i}': val_dice_per_class[i].item() for i in range(model.n_classes)},
            **{f'validation IoU class {i}': val_iou_per_class[i].item() for i in range(model.n_classes)},
            'epoch': epoch
        })

        logging.info(f"Epoch {epoch} - Validation Dice Score: {val_dice:.4f}, IoU: {val_iou:.4f}, Pixel Acc: {val_acc:.4f}")

        # Save the best model based on validation Dice Score
        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            run_name = wandb.run.name
            model_path = dir_checkpoint / f'best_model_{run_name}.pth'
            torch.save(model.state_dict(), str(model_path))
            logging.info(f'Best model saved as {model_path}!')

        # Optionally save checkpoint every epoch
        if save_checkpoint:
            checkpoint_path = dir_checkpoint / f'checkpoint_epoch{epoch}.pth'
            torch.save(model.state_dict(), str(checkpoint_path))
            logging.info(f'Checkpoint saved at {checkpoint_path}')
            
        
        
    # After all epochs are completed, evaluate on the test set
    logging.info("Training complete. Evaluating on test set...")

    # Load the best saved model
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = SegmentationDataset(dir_test_img, dir_test_mask, transform=None)  # Use same preprocessing as training
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate on the test set
    test_dice, test_iou, test_acc, test_dice_per_class, test_iou_per_class = evaluate(model, test_loader, device, amp=False)

    # Print test results
    logging.info(f"Test Dice Score (Mean): {test_dice:.4f}")
    logging.info(f"Test IoU (Mean): {test_iou:.4f}")
    logging.info(f"Test Pixel Accuracy: {test_acc:.4f}")

    for i, (dice, iou) in enumerate(zip(test_dice_per_class, test_iou_per_class)):
        logging.info(f"Class {i} - Dice: {dice:.4f}, IoU: {iou:.4f}")

    # Save results to a file
    with open(str(model_path).replace('best_model', 'test_results').replace('.pth', '.txt'), "w") as f:
        f.write(f"Test Dice Score (Mean): {test_dice:.4f}\n")
        f.write(f"Test IoU (Mean): {test_iou:.4f}\n")
        f.write(f"Test Pixel Accuracy: {test_acc:.4f}\n")
        for i, (dice, iou) in enumerate(zip(test_dice_per_class, test_iou_per_class)):
            f.write(f"Class {i} - Dice: {dice:.4f}, IoU: {iou:.4f}\n")

    logging.info(f"Test evaluation complete. Results saved to {str(model_path).replace('best_model', 'test_results').replace('.pth', '.txt')}")

    # Optional: Log test results to wandb
    experiment.log({
        "test Dice (avg)": test_dice,
        "test IoU (avg)": test_iou,
        "test Pixel Accuracy": test_acc,
        **{f"test Dice class {i}": test_dice_per_class[i].item() for i in range(model.n_classes)},
        **{f"test IoU class {i}": test_iou_per_class[i].item() for i in range(model.n_classes)},
    })
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', '-o', type=str, choices=['adamw', 'adam', 'rmsprop', 'sgd'], default='adamw', help='Choose optimizer')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last) # Tells PyTorch to store tensors in a format optimized for GPU efficiency

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
