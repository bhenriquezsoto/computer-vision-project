import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_loading import load_image, preprocessing
from models.unet_model import UNet
from models.autoencoder_model import Autoencoder
from models.unet_model import PointUNet

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

def predict_img(net,
                filename,
                device,
                dim=256,
                out_threshold=0.5):
    net.eval()
    
    # If model is Autoencoder, make sure we're in segmentation phase
    if isinstance(net, Autoencoder):
        net.set_phase("segmentation")
    
    full_img = load_image(filename)
    img, _ = preprocessing(img=full_img, mask=None, dim=dim)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # print("output shape", output.shape)
        # print("output:", output)
        output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
        if net.n_classes > 1:
            # probs = torch.softmax(output, dim=1)
            mask = torch.argmax(output, dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    print('mask unique values (should be just cast)', np.unique(mask))

    return mask[0].long().squeeze().numpy()

def predict_point_img(net,
                    filename,
                    point_coords,  # (y, x) coordinates of the clicked point
                    device,
                    dim=256,
                    sigma=10.0):
    """Predict segmentation mask for an image given a point prompt.
    
    Args:
        net: PointUNet model
        filename: Path to the image file
        point_coords: Tuple of (y, x) coordinates of the clicked point
        device: Device to run prediction on
        dim: Target image dimension
        sigma: Standard deviation for Gaussian heatmap
    """
    net.eval()
    
    # Load and preprocess image
    full_img = load_image(filename)
    img, _ = preprocessing(img=full_img, mask=None, dim=dim)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    # Generate point heatmap
    y, x = point_coords
    # Scale coordinates to match preprocessed image size
    scale_y = dim / full_img.shape[0]
    scale_x = dim / full_img.shape[1]
    scaled_y = int(y * scale_y)
    scaled_x = int(x * scale_x)
    
    # Create heatmap
    heatmap = np.zeros((dim, dim), dtype=np.float32)
    y_coords = np.arange(0, dim, 1, float)
    x_coords = np.arange(0, dim, 1, float)
    y_coords, x_coords = np.meshgrid(y_coords, x_coords)
    heatmap = np.exp(-((x_coords - scaled_x) ** 2 + (y_coords - scaled_y) ** 2) / (2 * sigma ** 2))
    
    # Convert heatmap to tensor
    heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_tensor = heatmap_tensor.to(device=device)

    with torch.no_grad():
        output = net(img, heatmap_tensor)
        # Use softmax for multi-class prediction
        output = F.softmax(output, dim=1)  
        output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
        mask = torch.argmax(output, dim=1)  # Get class with highest probability

    return mask[0].cpu().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--point', '-p', type=int, nargs=2, help='Point coordinates (y x) for point-based segmentation')
    parser.add_argument('--out-dir', '-d', type=str, default='src/a_unet/preds/', help='Directory to store output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--model-type', '-mt', type=str, choices=['unet', 'point_unet'], default='unet',
                        help='Type of model to use')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

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


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    if args.model_type == 'point_unet':
        if args.point is None:
            raise ValueError("Point coordinates are required for point-based segmentation")
        net = PointUNet(n_classes=3, bilinear=args.bilinear)  # 3 classes: background, cat, dog
    else:
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    
    # Load model weights and get mask values
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        net.load_state_dict(state_dict['model_state_dict'])
        mask_values = state_dict.get('mask_values', [0, 1, 2])  # Default to [0,1,2] if not found
    else:
        net.load_state_dict(state_dict)
        mask_values = [0, 1, 2]  # Default mask values for 3 classes
    
    logging.info('Model loaded!')
    logging.info(f'Using mask values: {mask_values}')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')

        if args.model_type == 'point_unet':
            mask = predict_point_img(
                net=net,
                filename=filename,
                point_coords=tuple(args.point),
                device=device,
                dim=args.img_dim
            )
        else:
            mask = predict_img(
                net=net,
                filename=filename,
                dim=args.img_dim,
                out_threshold=args.mask_threshold,
                device=device
            )

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)  # Use loaded mask values
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(Image.open(filename), mask)