#!/usr/bin/env python
"""
Simple UI for point-based segmentation.

This script provides a user interface where the user can:
1. Load an image
2. Click on the image to set a point prompt
3. Run the segmentation model to get the mask
4. Save the result

Usage:
    python point_segmentation_ui.py --model MODEL_PATH [--model-type MODEL_TYPE] [--classes NUM_CLASSES]
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, FileChooserWidget

from models.unet_model import PointUNet, UNet
from models.clip_model import CLIPSegmentationModel
from models.autoencoder_model import Autoencoder
from data_loading import load_image, preprocessing

class PointSegmentationUI:
    def __init__(self, model, device, img_dim=256, n_classes=3, mask_values=None):
        self.model = model
        self.device = device
        self.img_dim = img_dim
        self.n_classes = n_classes
        self.mask_values = mask_values or list(range(n_classes))
        
        self.img_path = None
        self.img = None
        self.full_img = None
        self.mask = None
        self.point = None
        self.sigma = 10  # Sigma for Gaussian heatmap
        
        # Set up the figure
        self.fig = plt.figure(figsize=(16, 9))
        self.setup_ui()
        
    def setup_ui(self):
        # Create a grid layout
        gs = self.fig.add_gridspec(2, 3)
        
        # Image panel
        self.ax_img = self.fig.add_subplot(gs[:, 0:2])
        self.ax_img.set_title('Input Image (Click to set a point)')
        self.ax_img.set_xlabel('No image loaded')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        
        # Mask panel
        self.ax_mask = self.fig.add_subplot(gs[0, 2])
        self.ax_mask.set_title('Segmentation Mask')
        self.ax_mask.set_xticks([])
        self.ax_mask.set_yticks([])
        
        # Controls panel
        self.ax_controls = self.fig.add_subplot(gs[1, 2])
        self.ax_controls.set_title('Controls')
        self.ax_controls.set_xticks([])
        self.ax_controls.set_yticks([])
        
        # Load image button
        load_button_ax = plt.axes([0.7, 0.4, 0.2, 0.05])
        self.load_button = Button(load_button_ax, 'Load Image')
        self.load_button.on_clicked(self.load_image)
        
        # Run segmentation button
        run_button_ax = plt.axes([0.7, 0.3, 0.2, 0.05])
        self.run_button = Button(run_button_ax, 'Run Segmentation')
        self.run_button.on_clicked(self.run_segmentation)
        
        # Save mask button
        save_button_ax = plt.axes([0.7, 0.2, 0.2, 0.05])
        self.save_button = Button(save_button_ax, 'Save Mask')
        self.save_button.on_clicked(self.save_mask)
        
        # Status text
        self.ax_controls.text(0.5, 0.1, 'Ready', ha='center', va='center', transform=self.ax_controls.transAxes)
        
        # Connect click event to image
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
    
    def load_image(self, event):
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        
        # Create Tk root
        root = Tk()
        root.withdraw()  # Hide the main window
        
        # Open file dialog
        file_path = askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.img_path = file_path
        self.load_image_from_path(file_path)
        
    def load_image_from_path(self, path):
        try:
            self.full_img = load_image(path)
            self.img, _ = preprocessing(img=self.full_img, mask=None, dim=self.img_dim, mode='valTest')
            
            # Display the image
            self.ax_img.clear()
            self.ax_img.imshow(self.full_img)
            self.ax_img.set_title('Input Image (Click to set a point)')
            self.ax_img.set_xlabel(os.path.basename(path))
            
            # Clear point and mask
            self.point = None
            self.mask = None
            
            # Clear mask display
            self.ax_mask.clear()
            self.ax_mask.set_title('Segmentation Mask')
            self.ax_mask.set_xticks([])
            self.ax_mask.set_yticks([])
            
            # Update status
            self.update_status('Image loaded. Click to set a point.')
            
            plt.draw()
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            logging.error(f"Error loading image: {str(e)}")
    
    def on_click(self, event):
        if event.inaxes == self.ax_img and self.full_img is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.point = (x, y)
            
            # Update the image with the point marker
            self.ax_img.clear()
            self.ax_img.imshow(self.full_img)
            self.ax_img.plot(x, y, 'ro', markersize=8)
            self.ax_img.set_title('Input Image (Point set)')
            self.ax_img.set_xlabel(os.path.basename(self.img_path) if self.img_path else '')
            
            # Update status
            self.update_status(f"Point set at ({x}, {y}). Click 'Run Segmentation'.")
            
            plt.draw()
    
    def run_segmentation(self, event):
        if self.full_img is None:
            self.update_status("Please load an image first.")
            return
        
        if self.point is None:
            self.update_status("Please set a point first.")
            return
        
        try:
            # Update status
            self.update_status("Running segmentation...")
            
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
                                        size=(self.img_dim, self.img_dim), 
                                        mode='bilinear', 
                                        align_corners=False).squeeze(0)
            
            # Run prediction
            self.model.eval()
            img_tensor = self.img.unsqueeze(0).to(self.device, dtype=torch.float32)
            heatmap_tensor = heatmap_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor, heatmap_tensor).cpu()
                output = F.interpolate(output, (height, width), mode='bilinear')
                
                if self.model.n_classes > 1:
                    self.mask = torch.argmax(output, dim=1)[0].numpy()
                else:
                    self.mask = (torch.sigmoid(output) > 0.5)[0, 0].numpy()
            
            # Display the mask
            self.ax_mask.clear()
            self.ax_mask.set_title('Segmentation Mask')
            
            # Visualize multi-class mask with different colors
            mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
            colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]  # Background, cat, dog
            
            for i, color in enumerate(colors[:self.n_classes]):
                mask_viz[self.mask == i] = color
            
            self.ax_mask.imshow(mask_viz)
            self.ax_mask.set_xticks([])
            self.ax_mask.set_yticks([])
            
            # Update status
            self.update_status(f"Segmentation complete. Class counts: {[np.sum(self.mask == i) for i in range(self.n_classes)]}")
            
            plt.draw()
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            logging.error(f"Error running segmentation: {str(e)}")
    
    def save_mask(self, event):
        if self.mask is None:
            self.update_status("No mask to save. Run segmentation first.")
            return
        
        try:
            from tkinter import Tk
            from tkinter.filedialog import asksaveasfilename
            
            # Create Tk root
            root = Tk()
            root.withdraw()  # Hide the main window
            
            # Default filename based on input image
            default_name = os.path.splitext(os.path.basename(self.img_path))[0] + "_mask.png" if self.img_path else "mask.png"
            
            # Open save dialog
            save_path = asksaveasfilename(
                title="Save segmentation mask",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                ]
            )
            
            if not save_path:
                return
            
            # Create RGB visualization of the mask
            height, width = self.mask.shape
            mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
            colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]  # Background, cat, dog
            
            for i, color in enumerate(colors[:self.n_classes]):
                mask_viz[self.mask == i] = color
            
            # Save the visualization
            cv2.imwrite(save_path, cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
            
            # Update status
            self.update_status(f"Mask saved to {save_path}")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            logging.error(f"Error saving mask: {str(e)}")
    
    def update_status(self, message):
        self.ax_controls.texts = []  # Clear existing text
        self.ax_controls.text(0.5, 0.1, message, ha='center', va='center', wrap=True, transform=self.ax_controls.transAxes)
        plt.draw()

def get_args():
    parser = argparse.ArgumentParser(description='Point Segmentation UI')
    parser.add_argument('--model', '-m', required=True, help='Path to the model file')
    parser.add_argument('--model-type', '-t', default='point_unet', choices=['point_unet', 'unet'], 
                      help='Type of model to use')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--img-dim', '-s', type=int, default=256, help='Image dimension')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse arguments
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Initialize model
    if args.model_type == 'point_unet':
        model = PointUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        logging.error(f"Model type '{args.model_type}' is not a point-based model")
        sys.exit(1)
    
    # Load model weights
    try:
        state_dict = torch.load(args.model, map_location=device)
        
        # Handle different state dict formats
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            mask_values = state_dict.get('mask_values', list(range(args.classes)))
        else:
            model.load_state_dict(state_dict)
            mask_values = list(range(args.classes))
        
        model.to(device)
        logging.info(f'Model loaded from {args.model}')
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Start UI
    ui = PointSegmentationUI(model, device, args.img_dim, args.classes, mask_values)
    plt.show() 