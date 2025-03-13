import argparse
import logging
import os
from pathlib import Path
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import wandb
from metrics import compute_metrics, compute_dice_per_class, compute_iou_per_class, compute_pixel_accuracy, dice_loss
from metrics import sigmoid_adaptive_focal_loss, adaptive_focal_loss_multiclass
from test import evaluate_model
from models.unet_model import UNet, PointUNet
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder
from data_loading import SegmentationDataset, TestSegmentationDataset, sort_and_match_files, PointSegmentationDataset, TestPointSegmentationDataset

dir_img = Path('Dataset/TrainVal/color')
dir_mask = Path('Dataset/TrainVal/label')
dir_test_img = Path('Dataset/Test/color')
dir_test_mask = Path('Dataset/Test/label')
dir_checkpoint = Path('src/models/checkpoints/')

def get_model(args):
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'clip':
        model = CLIPSegmentationModel(n_classes=args.classes)
    elif args.model == 'autoencoder':
        model = Autoencoder(n_channels=3, n_classes=args.classes)
    elif args.model == 'point_unet':
        model = PointUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    model = model.to(memory_format=torch.channels_last)
    return model

def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 16,
        optimizer: str = 'adam',
        class_weights: torch.Tensor = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = False,
        img_dim: int = 256,
        amp: bool = False,
        gradient_clipping: float = 1.0
):
    # Create checkpoint directory at the beginning to avoid repeated checks
    os.makedirs(dir_checkpoint, exist_ok=True)
    
    # 1. Get all image and mask files
    all_images = list(dir_img.glob('*'))
    all_masks = list(dir_mask.glob('*'))
    
    # Match images and masks by their base names using the new helper function
    matched_images, matched_masks = sort_and_match_files(all_images, all_masks)
    
    # Now split the matched pairs
    train_indices, val_indices = train_test_split(range(len(matched_images)), test_size=val_percent, random_state=42)
    
    train_images = [matched_images[i] for i in train_indices]
    train_masks = [matched_masks[i] for i in train_indices]
    val_images = [matched_images[i] for i in val_indices]
    val_masks = [matched_masks[i] for i in val_indices]
    
    # 2. Create dataset based on model type
    is_point_model = hasattr(model, 'is_point_model') and model.is_point_model
    
    if is_point_model:
        train_set = PointSegmentationDataset(train_images, train_masks, dim=img_dim, class_weights=class_weights.cpu().numpy().tolist() if class_weights is not None else None)
        val_set = TestPointSegmentationDataset(val_images, val_masks, dim=img_dim, class_weights=class_weights.cpu().numpy().tolist() if class_weights is not None else None)
    else:
        train_set = SegmentationDataset(train_images, train_masks, dim=img_dim)
        val_set = TestSegmentationDataset(val_images, val_masks, dim=img_dim)
    
    print("Training set dimensions: ", len(train_set))
    print("Validation set dimensions: ", len(val_set))

    # 3. Create data loaders
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_dim=img_dim, model=model.__class__.__name__, 
             amp=amp, optimizer=optimizer, dropout=0, is_point_model=is_point_model)
    )

    logging.info(f'''Starting training:
        Model:           {model.__class__.__name__} 
        Point model:     {is_point_model}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Weight decay:    {weight_decay}
        Optimizer:       {optimizer}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Image dimensions:{img_dim}x{img_dim}
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

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    grad_scaler = GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_val_iou = 0
    best_val_iou_after_epoch_10 = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        
        total_dice = torch.zeros(model.n_classes, device=device)  # Store per-class Dice
        total_iou = torch.zeros(model.n_classes, device=device)   # Store per-class IoU
        total_acc = 0  
        
        # Track number of batches each class appears in for proper averaging
        class_batch_count = torch.zeros(model.n_classes, device=device, dtype=torch.float32)

        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img', leave=True) as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == (model.n_image_channels if is_point_model else model.n_channels), \
                    f'Network has {model.n_channels} input channels, but got {images.shape[1]}.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Special handling for autoencoder model with two-phase training
                    if isinstance(model, Autoencoder):
                        # Phase 1: Reconstruction training
                        if epoch <= epochs // 2:
                            model.set_phase("reconstruction")
                            masks_pred = model(images)
                            # Reconstruction loss (MSE between input and output)
                            loss = nn.MSELoss()(masks_pred, images)
                        # Phase 2: Segmentation training
                        else:
                            # If this is the first epoch of segmentation phase, load pretrained encoder
                            if epoch == epochs // 2 + 1:
                                logging.info("Switching to segmentation phase, using pretrained encoder")
                                model.set_phase("segmentation")
                            
                            masks_pred = model(images)
                            # Segmentation loss with class weights
                            true_masks_processed = true_masks.clone()
                            true_masks_processed[true_masks_processed == 255] = 0  # Ignore void label
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(masks_pred, true_masks_processed, n_classes=model.n_classes)
                    elif is_point_model:
                        # For point-based models, pass both the image and point heatmap
                        point_heatmap = batch['point'].to(device=device, dtype=torch.float32)
                        masks_pred = model(images, point_heatmap)
                        
                        if model.n_classes == 1:
                            # Use Adaptive Focal Loss for binary segmentation
                            loss = sigmoid_adaptive_focal_loss(
                                masks_pred.squeeze(1), true_masks.float(), 
                                num_masks=images.shape[0],
                                epsilon=0.5, gamma=2.0, delta=0.4, alpha=1.0
                            )
                            # Can still add dice loss as a complementary loss
                            loss += dice_loss(masks_pred.squeeze(1), true_masks.float(), n_classes=model.n_classes)
                        else:
                            # Use Adaptive Focal Loss for multi-class segmentation
                            loss = adaptive_focal_loss_multiclass(
                                masks_pred, true_masks, 
                                num_masks=images.shape[0],
                                class_weights=class_weights,  # Pass the class weights here
                                epsilon=0.5, gamma=2.0, delta=0.4, alpha=1.0
                            )
                            
                            # Process targets for dice loss, treating void as background
                            true_masks_processed = true_masks.clone()
                            true_masks_processed[true_masks_processed == 255] = 0
                            
                            # Add dice loss as a complementary loss with a higher weight for better class balancing
                            loss += 1.2 * dice_loss(masks_pred, true_masks_processed, n_classes=model.n_classes)
                    else:
                        # Standard training for other models
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
                
                # Skip metrics during reconstruction phase of autoencoder
                if not (isinstance(model, Autoencoder) and model.training_phase == "reconstruction"):
                    # Handle void labels
                    true_masks_for_metrics = true_masks.clone()
                    true_masks_for_metrics[true_masks_for_metrics == 255] = 0  # Treat void label as background
                    
                    # Calculate metrics
                    dice_scores = compute_dice_per_class(masks_pred.argmax(dim=1), true_masks_for_metrics, n_classes=model.n_classes)
                    iou_scores = compute_iou_per_class(masks_pred.argmax(dim=1), true_masks_for_metrics, n_classes=model.n_classes)
                    pixel_acc = compute_pixel_accuracy(masks_pred.argmax(dim=1), true_masks_for_metrics)

                    # Check if each class is present in this batch (to avoid skewing metrics)
                    class_present = torch.zeros(model.n_classes, device=device, dtype=torch.bool)
                    for cls in range(model.n_classes):
                        class_present[cls] = (true_masks_for_metrics == cls).any()
                    
                    # Only accumulate metrics for classes that are present
                    total_dice += torch.where(class_present, dice_scores, torch.zeros_like(dice_scores))
                    total_iou += torch.where(class_present, iou_scores, torch.zeros_like(iou_scores))
                    
                    # Track number of batches each class appears in for proper averaging
                    class_batch_count += class_present.float()
                    
                    total_acc += pixel_acc

                    # Log training metrics
                    metrics_log = {
                        'train loss': loss.item(),
                        'train Dice (avg)': dice_scores.mean().item(),
                        'train IoU (avg)': iou_scores.mean().item(),
                        'train Pixel Accuracy': pixel_acc,
                        'step': global_step,
                        'epoch': epoch
                    }
                    
                    # Add per-class metrics only for classes present in the batch
                    for i in range(model.n_classes):
                        if class_present[i]:
                            metrics_log[f'train Dice class {i}'] = dice_scores[i].item()
                            metrics_log[f'train IoU class {i}'] = iou_scores[i].item()
                    
                    experiment.log(metrics_log)
                    
                    pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice_scores.mean().item():.4f}", iou=f"{iou_scores.mean().item():.4f}")
                else:
                    # During reconstruction phase, only log reconstruction loss
                    experiment.log({
                        'train reconstruction loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    
                    pbar.set_postfix(recon_loss=f"{loss.item():.4f}")

        # Compute average training metrics
        if not (isinstance(model, Autoencoder) and model.training_phase == "reconstruction"):
            # Avoid division by zero for classes that never appeared in any batch
            class_batch_count = torch.maximum(class_batch_count, torch.ones_like(class_batch_count))
            
            # Calculate proper per-class averages - no need for unsqueeze
            avg_dice = total_dice / class_batch_count
            avg_iou = total_iou / class_batch_count
            avg_acc = total_acc / len(train_loader)
            epoch_loss /= len(train_loader)

            logging.info(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}, Dice Score: {avg_dice.mean().item():.4f}, IoU: {avg_iou.mean().item():.4f}, Pixel Acc: {avg_acc:.4f}")
            
            # Log per-class metrics
            for i in range(model.n_classes):
                logging.info(f"  Class {i} - Dice: {avg_dice[i].item():.4f}, IoU: {avg_iou[i].item():.4f}")

            # Perform validation at the end of each epoch
            val_dice, val_iou, val_acc, val_dice_per_class, val_iou_per_class = compute_metrics(
                model, val_loader, device, amp, dim=img_dim, n_classes=model.n_classes, is_point_model=is_point_model
            )
            
            # Update scheduler
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

            # Save the best model based on validation IoU
            if val_iou > best_val_iou and epoch <= 10:
                best_val_iou = val_iou
                run_name = wandb.run.name
                model_path = os.path.join(dir_checkpoint, f'best_model_{run_name}.pth')
                state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
                torch.save(state_dict, model_path)
                logging.info(f'Best model saved as {model_path}!')
                
            # Save the best model based on validation IoU after epoch 10
            if epoch > 10 and val_iou > best_val_iou_after_epoch_10:
                best_val_iou_after_epoch_10 = val_iou
                run_name = wandb.run.name
                model_path = os.path.join(dir_checkpoint, f'best_model_after_epoch_10_{run_name}.pth')
                state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
                torch.save(state_dict, model_path)
                logging.info(f'Best model after epoch 10 saved as {model_path}!')
        else:
            # For reconstruction phase, only log reconstruction loss
            epoch_loss /= len(train_loader)
            logging.info(f"Epoch {epoch} - Reconstruction Loss: {epoch_loss:.4f}")
            
            # Save checkpoint after last reconstruction epoch
            if epoch == epochs // 2:
                run_name = wandb.run.name
                model_path = os.path.join(dir_checkpoint, f'reconstruction_model_{run_name}.pth')
                state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
                torch.save(state_dict, model_path)
                logging.info(f'Reconstruction model saved as {model_path}!')
                
            # Still update scheduler
            scheduler.step()

    # After all epochs are completed, evaluate on the test set
    logging.info("Training complete. Evaluating on test set...")
    
    # Load the best saved model
    state_dict = torch.load(model_path, map_location=device)
    # Check which key format is used in the checkpoint
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    logging.info(f'Model loaded from {model_path}')
    model.to(device)
    
    # Use the evaluate_model function from test.py
    test_dice, test_iou, test_acc, test_dice_per_class, test_iou_per_class = evaluate_model(
        model=model,
        device=device,
        img_dim=img_dim,
        amp=amp,
        n_classes=model.n_classes if hasattr(model, 'n_classes') else 3,
        results_path=None,  # No need to save results to a file as we log to wandb
        model_path=model_path,
        in_training=True,
        is_point_model=is_point_model
    )
    
    # Log test results to wandb
    if test_dice is not None:  # Make sure evaluation was successful
        experiment.log({
            "test Dice (avg)": test_dice,
            "test IoU (avg)": test_iou,
            "test Pixel Accuracy": test_acc,
            **{f"test Dice class {i}": test_dice_per_class[i].item() for i in range(model.n_classes if hasattr(model, 'n_classes') else 3)},
            **{f"test IoU class {i}": test_iou_per_class[i].item() for i in range(model.n_classes if hasattr(model, 'n_classes') else 3)},
        })
        
    experiment.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', '-o', type=str, choices=['adamw', 'adam', 'rmsprop', 'sgd'], default='adamw', help='Choose optimizer')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension'),
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'clip', 'autoencoder', 'point_unet'], default='unet', help='Choose model')
    parser.add_argument('--class-weights', '-cw', type=float, nargs='+', default=None, help='Class weights, space-separated (e.g., -cw 0.1 0.8 0.6)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = get_model(args)
    
    # Log different information based on model type
    if args.model == 'clip':
        logging.info(f'Network:\n'
                     f'\t{model.n_classes} output channels (classes)\n')
    elif args.model == 'autoencoder':
        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\tAutoencoder with two-phase training')
    elif args.model == 'point_unet':
        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\tPointUNet with two-phase training')
    else:  # UNet model
        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
        logging.info(f'Model loaded from {args.load}')
        
    # Set class weights
    if args.class_weights is not None:
        assert len(args.class_weights) == args.classes, \
            'Number of class weights must match number of classes. Expected: {} but got: {}'.format(args.classes, len(args.class_weights))
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None


    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            class_weights=class_weights,
            device=device,
            img_dim=args.img_dim,
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
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            class_weights=class_weights,
            device=device,
            img_dim=args.img_dim,
            val_percent=args.val / 100,
            amp=args.amp
        )
