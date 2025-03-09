import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data_loading import SegmentationDataset, load_image, preprocessing
from models.unet.unet_model import UNet
from models.clip.clip_model import CLIPSegmentationModel
from models.autoencoder.auto_encoder_model import AutoencoderSegmentation

import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_input_and_reconstruction(input_img, reconstruction):
    """Plot the input image and its reconstruction side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('Original Image')
    ax1.imshow(input_img)
    ax1.axis('off')
    
    ax2.set_title('Reconstructed Image')
    ax2.imshow(reconstruction)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_img(net,
                filename,
                device,
                dim=256,
                out_threshold=0.5,
                mode='segmentation'):
    net.eval()
    try:
        full_img = load_image(filename)
        img, _ = preprocessing(img=full_img, mask=None, dim=dim)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # Set the appropriate mode for autoencoder
            if hasattr(net, 'set_segmentation_mode') and hasattr(net, 'set_reconstruction_mode'):
                if mode == 'segmentation':
                    net.set_segmentation_mode()
                else:
                    net.set_reconstruction_mode()
                    
            # Run forward pass
            output = net(img).cpu()
            
            # For segmentation or standard models
            if mode == 'segmentation' or not hasattr(net, 'mode'):
                output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
                if net.n_classes > 1:
                    mask = torch.argmax(output, dim=1)
                else:
                    mask = torch.sigmoid(output) > out_threshold
                return mask[0].long().squeeze().numpy(), None
            else:
                # For reconstruction mode - return the reconstructed image
                output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
                # Clamp outputs to valid image range
                output = torch.clamp(output, 0, 1)
                return None, output[0].permute(1, 2, 0).numpy()  # Convert to HWC format
    except Exception as e:
        logging.error(f"Error predicting image: {str(e)}")
        raise


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model-path', '-m', default='MODEL.pth', metavar='FILE',
                        help='Path to the file containing model weights')
    parser.add_argument('--model-type', '-t', type=str, choices=['unet', 'clip', 'autoencoder'], default='unet',
                        help='Model type (unet, clip, or autoencoder)')
    parser.add_argument('--mode', type=str, choices=['segmentation', 'reconstruction'], default='segmentation',
                       help='Mode for autoencoder: segmentation or reconstruction')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--out-dir', '-d', type=str, default='src/models/preds/', help='Directory to store output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-th', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        suffix = '_RECON.png' if args.mode == 'reconstruction' else '_OUT.png'
        return f'{os.path.splitext(fn)[0]}{suffix}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    in_files = args.input
    out_files = get_output_filenames(args)

    # Create output directory if needed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create and initialize the model based on model type
    try:
        if args.model_type == 'unet':
            net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        elif args.model_type == 'clip':
            net = CLIPSegmentationModel(n_classes=args.classes)
        elif args.model_type == 'autoencoder':
            net = AutoencoderSegmentation(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}")
        raise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model from {args.model_path}')
    logging.info(f'Using device {device}')
    
    # Log prediction mode for autoencoder
    if args.model_type == 'autoencoder':
        logging.info(f'Autoencoder mode: {args.mode}')

    try:
        # Move model to device
        net.to(device=device)
        
        # Check if model file exists
        if not os.path.isfile(args.model_path):
            logging.error(f"Model file not found: {args.model_path}")
            return
            
        # Load model weights
        state_dict = torch.load(args.model_path, map_location=device)
        
        # Handle different state_dict formats
        if 'model_state_dict' in state_dict:
            net.load_state_dict(state_dict['model_state_dict'])
            mask_values = state_dict.get('mask_values', [0, 1, 2])  # Default to standard classes if not found
        else:
            # Try to load directly (older models)
            net.load_state_dict(state_dict)
            mask_values = [0, 1, 2]  # Default mask values
            
        logging.info('Model loaded!')
        if args.mode == 'segmentation':
            logging.info(f'Mask values: {mask_values}')

        # Ensure the output directory exists
        os.makedirs(args.out_dir, exist_ok=True)

        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} ...')
            
            try:
                mask, reconstruction = predict_img(
                    net=net,
                    filename=filename,
                    dim=args.img_dim,
                    out_threshold=args.mask_threshold,
                    device=device,
                    mode=args.mode
                )

                if not args.no_save:
                    out_filename = os.path.join(args.out_dir, os.path.basename(out_files[i]))
                    
                    if args.mode == 'segmentation' or not hasattr(net, 'mode'):
                        # Save segmentation mask
                        result = mask_to_image(mask, mask_values)
                        result.save(out_filename)
                        logging.info(f'Mask saved to {out_filename}')
                    else:
                        # Save reconstructed image
                        # Convert reconstruction to uint8 image
                        recon_img = (reconstruction * 255).astype(np.uint8)
                        Image.fromarray(recon_img).save(out_filename)
                        logging.info(f'Reconstruction saved to {out_filename}')

                if args.viz:
                    logging.info(f'Visualizing results for image {filename}, close to continue...')
                    if args.mode == 'segmentation' or not hasattr(net, 'mode'):
                        # Visualize segmentation result
                        plot_img_and_mask(Image.open(filename), mask)
                    else:
                        # Visualize reconstruction result
                        plot_input_and_reconstruction(np.asarray(Image.open(filename)), reconstruction)
            except Exception as e:
                logging.error(f"Error processing image {filename}: {str(e)}")
                continue
                
    except FileNotFoundError:
        logging.error(f"Model file not found: {args.model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


if __name__ == '__main__':
    main()