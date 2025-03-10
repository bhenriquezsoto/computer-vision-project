import argparse
import logging
import os
from pathlib import Path
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import wandb
from metrics import compute_dice_per_class, compute_iou_per_class, compute_pixel_accuracy, dice_loss
from models.unet_model import UNet
from data_loading import SegmentationDataset, TestSegmentationDataset


dir_img = Path('Dataset/TrainVal/color')
dir_mask = Path('Dataset/TrainVal/label')
dir_test_img = Path('Dataset/Test/color')
dir_test_mask = Path('Dataset/Test/label')
dir_checkpoint = Path('src/a_unet/checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 16,
        optimizer: str = 'adam',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = False,
        img_dim: int = 256,
        amp: bool = False,
        gradient_clipping: float = 1.0
):
    # 1. Split into train / validation partitions
    all_images = list(dir_img.glob('*'))
    all_masks = list(dir_mask.glob('*'))
    
    
    train_images, val_images, train_masks, val_masks = train_test_split(all_images, all_masks, test_size=val_percent, random_state=42)
    
    
    # 2. Create dataset. If augmentation is enabled, tune the augmentation parameters in 'data_loading.py'

    train_set = SegmentationDataset(train_images, train_masks, dim=img_dim)
    val_set = TestSegmentationDataset(val_images, val_masks, dim=img_dim)

    print("Training set dimensions: ", len(train_set))

    # 3. Create data loaders
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_dim=img_dim, amp=amp, optimizer=optimizer, dropout=0)
    )

    logging.info(f'''Starting training:
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

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # or ReduceLROnPlateau(optimizer, mode='max', patience=5)
    grad_scaler = GradScaler(enabled=amp)
    weight = torch.tensor([1.0, 2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weight) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
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
        
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img', leave=True) as pbar:
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
        val_dice, val_iou, val_acc, val_dice_per_class, val_iou_per_class = evaluate(model, val_loader, device, amp, dim=img_dim, n_classes=model.n_classes)
        
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
        if val_iou > best_val_iou and epoch <= 10:
            best_val_iou = val_iou
            run_name = wandb.run.name
            model_path = os.path.join(dir_checkpoint, f'best_model_{run_name}.pth')
            state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
            torch.save(state_dict, model_path)
            logging.info(f'Best model saved as {model_path}!')
            
        if epoch > 10 and val_iou > best_val_iou_after_epoch_10:
            best_val_iou_after_epoch_10 = val_iou
            run_name = wandb.run.name
            model_path = os.path.join(dir_checkpoint, f'best_model_after_epoch_10_{run_name}.pth')
            state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
            torch.save(state_dict, model_path)
            logging.info(f'Best model after epoch 10 saved as {model_path}!')

        # Optionally save checkpoint every epoch
        if save_checkpoint or epoch == epochs or epoch == 50:
            checkpoint_path = os.path.join(dir_checkpoint, f'checkpoint_epoch{epoch}.pth')
            state_dict = {"model_state_dict": model.state_dict(), "mask_values": train_set.mask_values}
            torch.save(state_dict, checkpoint_path)
            logging.info(f'Checkpoint saved at {checkpoint_path}')
            
        
        
    # After all epochs are completed, evaluate on the test set
    logging.info("Training complete. Evaluating on test set...")

    # Load the best saved model
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    logging.info(f'Model loaded from {model_path}')
    model.to(device)
    model.eval()

    # Load test dataset
    test_img_files = list(dir_test_img.glob('*'))
    test_mask_files = list(dir_test_mask.glob('*'))
    
    test_dataset = TestSegmentationDataset(test_img_files, test_mask_files, dim=img_dim)  # Use same preprocessing as training
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_args)

    # Evaluate on the test set
    test_dice, test_iou, test_acc, test_dice_per_class, test_iou_per_class = evaluate(model, test_loader, device, amp=amp, dim=img_dim, desc='Testing round')

    # Print test results
    logging.info(f"Test Dice Score (Mean): {test_dice:.4f}")
    logging.info(f"Test IoU (Mean): {test_iou:.4f}")
    logging.info(f"Test Pixel Accuracy: {test_acc:.4f}")

    for i, (dice, iou) in enumerate(zip(test_dice_per_class, test_iou_per_class)):
        logging.info(f"Class {i} - Dice: {dice:.4f}, IoU: {iou:.4f}")

    # Save results to a file
    with open(model_path.replace('best_model', 'test_results').replace('.pth', '.txt'), "w") as f:
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
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension'),
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
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
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
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
            device=device,
            img_dim=args.img_dim,
            val_percent=args.val / 100,
            amp=args.amp
        )
