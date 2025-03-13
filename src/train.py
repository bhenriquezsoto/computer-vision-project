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
import torch.nn.functional as F


import wandb
from metrics import compute_dice_per_class, compute_iou_per_class, compute_pixel_accuracy, dice_loss
from eval_utils import evaluate_segmentation
from test import evaluate_model
from models.unet_model import UNet, PointUNet 
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder
from data_loading import SegmentationDataset, TestSegmentationDataset, sort_and_match_files, PointSegmentationDataset, TestPointSegmentationDataset, calculate_class_weights
from losses import adaptive_focal_loss, adaptive_focal_loss_multi_class


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
    elif args.model == 'pointunet':
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
        gradient_clipping: float = 1.0,
        point_based: bool = False,
        sigma: float = 3.0
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
    
    # 2. Create dataset
    if point_based:
        train_set = PointSegmentationDataset(train_images, train_masks, dim=img_dim, sigma=sigma)
        val_set = TestPointSegmentationDataset(val_images, val_masks, dim=img_dim, sigma=sigma)
        is_point_model = True
    else:
        train_set = SegmentationDataset(train_images, train_masks, dim=img_dim)
        val_set = TestSegmentationDataset(val_images, val_masks, dim=img_dim)
        is_point_model = False
    
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
             amp=amp, optimizer=optimizer, dropout=0, point_based=point_based, point_sigma=sigma)
    )

    logging.info(f'''Starting training:
        Model:           {model.__class__.__name__} 
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
        Point-based:     {point_based}
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
    
    # Use Adaptive Focal Loss instead of CrossEntropyLoss
    criterion = adaptive_focal_loss_multi_class if model.n_classes > 1 else adaptive_focal_loss
    
    global_step = 0
    best_val_iou = 0
    best_val_iou_after_epoch_10 = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        total_dice = torch.zeros(model.n_classes, device=device)
        total_iou = torch.zeros(model.n_classes, device=device)
        total_acc = 0  
        
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img', leave=True) as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = batch['mask'].to(device=device, dtype=torch.long)

                # Special handling for point-based models
                if is_point_model:
                    points = batch['point'].to(device=device, dtype=torch.float32)
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        # Forward pass with additional point input
                        masks_pred = model(images, points)
                        
                        # Loss calculation using Adaptive Focal Loss with class weights
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float(), len(images))
                            loss += 0.5 * dice_loss(masks_pred.squeeze(1), true_masks.float(), n_classes=model.n_classes)
                        else:
                            # Use weighted combination of AFL and Dice loss
                            afl_loss = criterion(masks_pred, true_masks, len(images), class_weights=class_weights)
                            # For dice loss, we still need to exclude void pixels
                            mask_for_dice = true_masks.clone()
                            dice = dice_loss(masks_pred, mask_for_dice, n_classes=model.n_classes, ignore_index=255, class_weights=class_weights)
                            loss = afl_loss + 0.3 * dice  # Give more weight to AFL
                else:
                    # Standard forward pass for regular models
                    assert images.shape[1] == model.n_channels, \
                        f'Network has {model.n_channels} input channels, but got {images.shape[1]}.'

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        # Special handling for autoencoder model with two-phase training
                        if isinstance(model, Autoencoder):
                            # Phase 1: Reconstruction training (first half of epochs)
                            if epoch <= epochs // 2:
                                model.set_phase("reconstruction")
                                masks_pred = model(images)
                                # Reconstruction loss (MSE between input and output)
                                loss = nn.MSELoss()(masks_pred, images)
                            # Phase 2: Segmentation training (second half of epochs)
                            else:
                                # If this is the first epoch of segmentation phase, load pretrained encoder
                                if epoch == epochs // 2 + 1:
                                    logging.info("Switching to segmentation phase, using pretrained encoder")
                                    # Use current model state for encoder weights
                                    model.set_phase("segmentation")
                                
                                masks_pred = model(images)
                                # Segmentation loss with Adaptive Focal Loss and class weights
                                loss = criterion(masks_pred, true_masks, len(images), class_weights=class_weights)
                                # For dice loss, we still need to exclude void pixels
                                mask_for_dice = true_masks.clone()
                                loss += dice_loss(masks_pred, mask_for_dice, n_classes=model.n_classes, ignore_index=255, class_weights=class_weights)
                        else:
                            # Standard training for other models
                            masks_pred = model(images)
                            if model.n_classes == 1:
                                loss = criterion(masks_pred.squeeze(1), true_masks.float(), len(images))
                                loss += dice_loss(masks_pred.squeeze(1), true_masks.float(), n_classes=model.n_classes)
                            else:
                                loss = criterion(masks_pred, true_masks, len(images), class_weights=class_weights)
                                # For dice loss, we still need to exclude void pixels
                                mask_for_dice = true_masks.clone()
                                loss += dice_loss(masks_pred, mask_for_dice, n_classes=model.n_classes, ignore_index=255, class_weights=class_weights)

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
                # Skip these metrics during reconstruction phase of autoencoder
                if not (isinstance(model, Autoencoder) and model.training_phase == "reconstruction"):
                    # Get predictions for logging (not affecting training)
                    with torch.no_grad():
                        pred_mask = masks_pred.argmax(dim=1)
                        
                        # For metrics calculation, create a mask excluding void pixels (255)
                        metrics_mask = true_masks.clone()
                        void_pixels = metrics_mask == 255
                        
                        # Set void pixels to match prediction to exclude them from metric calculations
                        metrics_mask[void_pixels] = pred_mask[void_pixels]
                        
                        dice_scores = compute_dice_per_class(pred_mask, metrics_mask, n_classes=model.n_classes)
                        iou_scores = compute_iou_per_class(pred_mask, metrics_mask, n_classes=model.n_classes)
                        pixel_acc = compute_pixel_accuracy(pred_mask, metrics_mask)

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
            avg_dice = total_dice / len(train_loader)
            avg_iou = total_iou / len(train_loader)
            avg_acc = total_acc / len(train_loader)
            epoch_loss /= len(train_loader)

            logging.info(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}, Dice Score: {avg_dice.mean().item():.4f}, IoU: {avg_iou.mean().item():.4f}, Pixel Acc: {avg_acc:.4f}")

            # Perform validation at the end of each epoch
            val_dice, val_iou, val_acc, val_dice_per_class, val_iou_per_class = evaluate_segmentation(
                net=model,
                dataloader=val_loader,
                device=device,
                amp=amp,
                n_classes=model.n_classes,
                class_weights=class_weights,
                mode='val',
                is_point_model=is_point_model,
                desc="Validation round"
            )
            
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
            if val_iou > best_val_iou and epoch <= 10:
                best_val_iou = val_iou
                run_name = wandb.run.name
                model_path = os.path.join(dir_checkpoint, f'best_model_{run_name}.pth')
                state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
                torch.save(state_dict, model_path)
                logging.info(f'Best model saved as {model_path}!')
                
            # Save the best model based on validation IoU after epoch 10 to avoid only saving early peaks
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
    
    # Test evaluation with consistent approach for all model types
    test_images = list(dir_test_img.glob('*'))
    test_masks = list(dir_test_mask.glob('*'))
    test_images, test_masks = sort_and_match_files(test_images, test_masks)
    
    if is_point_model:
        test_dataset = TestPointSegmentationDataset(test_images, test_masks, dim=img_dim, sigma=sigma)
    else:
        test_dataset = TestSegmentationDataset(test_images, test_masks, dim=img_dim)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_args)
    
    # Unified evaluation approach for all model types
    test_dice, test_iou, test_acc, test_dice_per_class, test_iou_per_class = evaluate_segmentation(
        net=model,
        dataloader=test_loader,
        device=device,
        amp=amp,
        n_classes=model.n_classes if hasattr(model, 'n_classes') else 3,
        class_weights=class_weights,
        mode='test',
        is_point_model=is_point_model,
        desc="Test evaluation"
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
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'clip', 'autoencoder', 'pointunet'], default='unet', help='Choose model')
    parser.add_argument('--class-weights', '-cw', type=float, nargs='+', default=None, help='Class weights, space-separated (e.g., -cw 0.1 0.8 0.6)')
    parser.add_argument('--point-based', '-p', action='store_true', default=False, help='Use point-based segmentation')
    parser.add_argument('--point-sigma', '-ps', type=float, default=3.0, help='Sigma for Gaussian point heatmap')

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
    elif args.model == 'pointunet':
        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels + 1 point channel\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling\n'
                    f'\tPoint-based segmentation with sigma={args.point_sigma}')
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
        # Get all image and mask files and match them
        all_images = list(dir_img.glob('*'))
        all_masks = list(dir_mask.glob('*'))
        matched_images, matched_masks = sort_and_match_files(all_images, all_masks)
        
        # Calculate class weights based on dataset statistics
        logging.info("Calculating class weights from dataset statistics...")
        class_weights = calculate_class_weights(matched_masks, args.classes)
        class_weights = class_weights.to(device)
        logging.info(f"Calculated class weights: {class_weights.tolist()}")

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
            amp=args.amp,
            point_based=args.point_based,
            sigma=args.point_sigma
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
            amp=args.amp,
            point_based=args.point_based,
            sigma=args.point_sigma
        )
