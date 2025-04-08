import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

from data_loading import SegmentationDataset, load_image, preprocessing, generate_point_heatmap
from models.unet_model import UNet, PointUNet
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class PointPromptUI:
    """
    A simple UI for point-prompt segmentation.
    
    This allows users to click on an image to specify a point prompt, 
    and then the model will predict the segmentation mask.
    """
    def __init__(self, model, device, dim, mask_values, out_threshold=0.5):
        self.model = model
        self.device = device
        self.dim = dim
        self.mask_values = mask_values
        self.out_threshold = out_threshold
        self.point = None
        self.img = None
        self.full_img = None
        self.sigma = 10  # Sigma for Gaussian heatmap
        self.mask = None
        
    def setup_ui(self, img_path):
        # Load image
        self.full_img = load_image(img_path)
        img, _ = preprocessing(img=self.full_img, mask=None, dim=self.dim, mode='valTest')
        self.img = img
        
        # Set up the figure
        fig, (self.ax_img, self.ax_mask) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the image
        self.ax_img.imshow(self.full_img)
        self.ax_img.set_title('Click on the image to set a point prompt')
        self.ax_img.set_xlabel('Press "Run" to predict the segmentation')
        
        # Set up empty mask display
        self.ax_mask.set_title('Predicted mask will appear here')
        self.ax_mask.axis('off')
        
        # Add run button
        ax_button = plt.axes([0.45, 0.05, 0.1, 0.075])
        self.button = Button(ax_button, 'Run')
        self.button.on_clicked(self.run_prediction)
        
        # Connect click event
        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
    
    def on_click(self, event):
        if event.inaxes == self.ax_img:
            x, y = int(event.xdata), int(event.ydata)
            self.point = (x, y)
            
            # Update the image with a marker
            self.ax_img.clear()
            self.ax_img.imshow(self.full_img)
            self.ax_img.plot(x, y, 'ro', markersize=8)
            self.ax_img.set_title('Point prompt set')
            plt.draw()
    
    def run_prediction(self, event):
        if self.point is None:
            self.ax_mask.set_title('Please set a point prompt first')
            plt.draw()
            return
        
        # Generate heatmap from the point
        x, y = self.point
        height, width = self.full_img.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        heatmap[y, x] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        heatmap = heatmap / heatmap.max()
        
        # Resize heatmap to match the processed image size
        heatmap_tensor = torch.from_numpy(heatmap).float()
        heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0).unsqueeze(0), 
                                    size=(self.dim, self.dim), 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(0)
        
        # Run prediction
        logging.info('Running prediction...')
        self.model.eval()
        img_tensor = self.img.unsqueeze(0).to(self.device, dtype=torch.float32)
        heatmap_tensor = heatmap_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor, heatmap_tensor).cpu()
            output = F.interpolate(output, (height, width), mode='bilinear')
            
            if self.model.n_classes > 1:
                mask = torch.argmax(output, dim=1)[0].numpy()
            else:
                mask = (torch.sigmoid(output) > self.out_threshold)[0, 0].numpy()
        
        self.mask = mask
        
        # Display the mask
        self.ax_mask.clear()
        self.ax_mask.set_title('Predicted mask')
        
        # Visualize multi-class mask with different colors
        mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
        colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]  # Background, cat, dog
        
        for i, color in enumerate(colors[:self.model.n_classes]):
            mask_viz[mask == i] = color
        
        self.ax_mask.imshow(mask_viz)
        plt.draw()
        
        # Save the mask
        mask_img = mask_to_image(mask, self.mask_values)
        save_path = os.path.splitext(img_path)[0] + '_pred.png'
        mask_img.save(save_path)
        logging.info(f'Mask saved to {save_path}')

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
                out_threshold=0.5,
                point_coords=None):
    """
    Predict segmentation mask for an input image.
    
    Args:
        net: The model
        filename: Path to the input image
        device: Device to run on
        dim: Image dimension
        out_threshold: Threshold for binary segmentation
        point_coords: Optional (x, y) coordinates for point prompt
        
    Returns:
        np.ndarray: Predicted mask
    """
    net.eval()
    
    # If model is Autoencoder, make sure we're in segmentation phase
    if isinstance(net, Autoencoder):
        net.set_phase("segmentation")
    
    full_img = load_image(filename)
    img, _ = preprocessing(img=full_img, mask=None, dim=dim, mode='valTest')
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    is_point_model = hasattr(net, 'is_point_model') and net.is_point_model
    
    with torch.no_grad():
        if is_point_model:
            # Handle point-based model
            if point_coords is None:
                # If no point provided, use center of image
                height, width = full_img.shape[:2]
                point_coords = (width // 2, height // 2)
                logging.info(f"No point provided, using center point {point_coords}")
            
            # Generate heatmap from the point
            x, y = point_coords
            height, width = full_img.shape[:2]
            heatmap = np.zeros((height, width), dtype=np.float32)
            heatmap[y, x] = 1
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=10, sigmaY=10)
            heatmap = heatmap / heatmap.max()
            
            # Resize heatmap to match the processed image size
            heatmap_tensor = torch.from_numpy(heatmap).float()
            heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0).unsqueeze(0), 
                                         size=(dim, dim), 
                                         mode='bilinear', 
                                         align_corners=False).squeeze(0)
            
            # Move to device
            heatmap_tensor = heatmap_tensor.to(device)
            
            # Run prediction
            output = net(img, heatmap_tensor).cpu()
        else:
            # Standard model
            output = net(img).cpu()
        
        # Resize to original size
        output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
        
        if net.n_classes > 1:
            mask = torch.argmax(output, dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    
    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--out-dir', '-d', type=str, default='src/models/preds/', help='Directory to store output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--model-type', '-mt', type=str, choices=['unet', 'clip', 'autoencoder', 'point_unet'], 
                       default='unet', help='Type of model to use')
    parser.add_argument('--interactive', '-int', action='store_true', 
                       help='Run in interactive mode for point-based segmentation')
    parser.add_argument('--point', '-p', type=int, nargs=2, 
                       help='Specify point coordinates (x y) for point-based segmentation')
    
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

    # Initialize the model based on model type
    if args.model_type == 'unet':
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_type == 'clip':
        net = CLIPSegmentationModel(n_classes=args.classes)
    elif args.model_type == 'autoencoder':
        net = Autoencoder(n_channels=3, n_classes=args.classes)
        net.set_phase("segmentation")
    elif args.model_type == 'point_unet':
        net = PointUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    
    # Handle different state dict formats
    if 'model_state_dict' in state_dict:
        net.load_state_dict(state_dict['model_state_dict'])
        mask_values = state_dict['mask_values']
    else:
        net.load_state_dict(state_dict)
        # Default mask values if not provided
        mask_values = [0, 1, 2]  # Background, cat, dog
            
    logging.info('Model loaded!')
    logging.info(f'Mask values: {mask_values}')
    
    # Check if model is point-based and interactive mode is requested
    is_point_model = hasattr(net, 'is_point_model') and net.is_point_model
    
    if args.interactive and is_point_model:
        # Run interactive mode
        if len(in_files) != 1:
            logging.error('Interactive mode requires exactly one input image')
            sys.exit(1)
        
        logging.info('Running in interactive mode')
        ui = PointPromptUI(net, device, args.img_dim, mask_values, args.mask_threshold)
        ui.setup_ui(in_files[0])
    else:
        # Standard prediction mode
        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} ...')
            
            point_coords = args.point if is_point_model and args.point else None

            mask = predict_img(
                net=net,
                filename=filename,
                dim=args.img_dim,
                out_threshold=args.mask_threshold,
                device=device,
                point_coords=point_coords
            )

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(Image.open(filename), mask)