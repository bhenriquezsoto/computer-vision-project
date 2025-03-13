import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data_loading import SegmentationDataset, load_image, preprocessing, create_point_heatmap
from models.unet_model import UNet, PointUNet
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder

import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask, point=None):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    
    # If point is provided, plot it on the input image
    if point is not None:
        y, x = point
        ax[0].plot(x, y, 'ro', markersize=5)
        
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def predict_img(net,
                filename,
                device,
                dim=256,
                out_threshold=0.5,
                point=None,
                point_sigma=3.0):
    """
    Predict mask for an image using a trained model.
    
    Args:
        net: The trained model
        filename: Path to the input image
        device: Device to run prediction on
        dim: Image dimension for preprocessing
        out_threshold: Threshold for binary segmentation
        point: Optional tuple (y, x) for point-based segmentation
        point_sigma: Sigma value for Gaussian point heatmap
    
    Returns:
        numpy.ndarray: Predicted segmentation mask
    """
    net.eval()
    
    # If model is Autoencoder, make sure we're in segmentation phase
    if isinstance(net, Autoencoder):
        net.set_phase("segmentation")
    
    full_img = load_image(filename)
    img, _ = preprocessing(img=full_img, mask=None, dim=dim)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Check if it's a point-based model and point is provided
        if isinstance(net, PointUNet) and point is not None:
            # Convert point coordinates from original image space to processed space
            scale_y = dim / full_img.shape[0]
            scale_x = dim / full_img.shape[1]
            processed_y = min(int(point[0] * scale_y), dim - 1)
            processed_x = min(int(point[1] * scale_x), dim - 1)
            
            # Create point heatmap
            point_heatmap = create_point_heatmap((processed_y, processed_x), (dim, dim), sigma=point_sigma)
            point_heatmap_tensor = torch.from_numpy(point_heatmap).unsqueeze(0).unsqueeze(0)
            point_heatmap_tensor = point_heatmap_tensor.to(device=device, dtype=torch.float32)
            
            # Forward pass with image and point
            output = net(img, point_heatmap_tensor).cpu()
        else:
            # Regular forward pass for standard models
            output = net(img).cpu()
            
        # Resize output to match original image size
        output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
        
        if net.n_classes > 1:
            # Multi-class segmentation
            mask = torch.argmax(output, dim=1)
        else:
            # Binary segmentation
            mask = torch.sigmoid(output) > out_threshold
    
    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--out-dir', '-d', type=str, default='src/a_unet/preds/', help='Directory to store output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--model-type', '-mt', type=str, choices=['unet', 'pointunet', 'clip', 'autoencoder'], default='unet',
                        help='Type of model to use')
    parser.add_argument('--point', '-p', type=int, nargs=2, default=None, 
                        help='Point coordinates (y x) for point-based segmentation')
    parser.add_argument('--point-sigma', '-ps', type=float, default=3.0, 
                        help='Sigma for Gaussian point heatmap')
    
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

    # Initialize model based on type
    if args.model_type == 'unet':
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_type == 'pointunet':
        net = PointUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        if args.point is None:
            logging.warning("No point provided for point-based segmentation. Using center point.")
            args.point = (args.img_dim // 2, args.img_dim // 2)
    elif args.model_type == 'clip':
        net = CLIPSegmentationModel(n_classes=args.classes)
    elif args.model_type == 'autoencoder':
        net = Autoencoder(n_channels=3, n_classes=args.classes)
        net.set_phase("segmentation")  # Ensure we're in segmentation phase
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    net.load_state_dict(state_dict['model_state_dict'])
    mask_values = state_dict['mask_values']

    logging.info('Model loaded!')
    logging.info(f'Mask values: {mask_values}')

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')

        mask = predict_img(net=net,
                           filename=filename,
                           dim=args.img_dim,
                           out_threshold=args.mask_threshold,
                           device=device,
                           point=args.point,
                           point_sigma=args.point_sigma)

        if not args.no_save:
            out_filename = os.path.join(args.out_dir, os.path.basename(out_files[i]))
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            input_img = np.array(Image.open(filename))
            plot_img_and_mask(input_img, mask, args.point if args.model_type == 'pointunet' else None)