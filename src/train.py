import argparse
import logging
from pathlib import Path
import torch

from.models.autoencoder.autoencoder_train import train_autoencoder_model
from segmentation_train import train_segmentation_model

# import models
from models.unet.unet_model import UNet
from models.clip.clip_model import CLIPSegmentationModel
from models.autoencoder.autoencoder_model import AutoencoderSegmentation

dir_img = Path('Dataset/TrainVal/color')
dir_mask = Path('Dataset/TrainVal/label')
dir_test_img = Path('Dataset/Test/color')
dir_test_mask = Path('Dataset/Test/label')
dir_checkpoint = Path('src/checkpoints/')

def get_model(args):
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        
    elif args.model == 'clip':
        model = CLIPSegmentationModel(n_classes=args.classes)
        
    elif args.model == 'autoencoder':
        model = AutoencoderSegmentation(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        model.set_reconstruction_mode()
        
    elif args.model == 'autoencoder_seg':
        model = AutoencoderSegmentation(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        try:
            model.load_state_dict(torch.load('src/a_unet/autoencoder.pth')['model_state_dict'])
        except FileNotFoundError:
            logging.error('Autoencoder weights not found. Ensure that the autoencoder.pth file is present in the src/a_unet directory')
        model.set_segmentation_mode()
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    model = model.to(memory_format=torch.channels_last)
    return model


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
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'clip', 'autoencoder', 'autoencoder_seg'], default='unet', help='Choose model (unet, clip, autoencoder, autoencoder_seg)')
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
    
    # Log model architecture
    if args.model == 'clip':
        logging.info(f'Network:\n'
                     f'\t{model.n_classes} output channels (classes)\n')
    else:
        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Load model from a .pth file if specified
    if args.load:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
        logging.info(f'Model loaded from {args.load}')
        
    # Set class weights
    if args.class_weights:
        assert len(args.class_weights) == args.classes, \
            'Number of class weights must match number of classes. Expected: {} but got: {}'.format(args.classes, len(args.class_weights))
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None
        
    # Set up optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.999)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.999)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        

    model.to(device=device)
    try:
        if args.model == 'autoencoder':
            train_autoencoder_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                optimizer=optimizer,
                device=device,
                img_dim=args.img_dim,
                val_percent=args.val / 100,
                amp=args.amp
            )
        else:
            train_segmentation_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                optimizer=optimizer,
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
        if args.model == 'autoencoder':
            train_autoencoder_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                optimizer=optimizer,
                device=device,
                img_dim=args.img_dim,
                val_percent=args.val / 100,
                amp=args.amp
            )
        else:
            train_segmentation_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                optimizer=optimizer,
                class_weights=class_weights,
                device=device,
                img_dim=args.img_dim,
                val_percent=args.val / 100,
                amp=args.amp
            )
