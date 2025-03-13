import gradio as gr
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
import os
from pathlib import Path

from models.unet_model import PointUNet
from data_loading import load_image, preprocessing

def load_model(model_path, device):
    """Load the point segmentation model"""
    model = PointUNet(n_channels=3, n_classes=3, bilinear=False)
    model.to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    return model

def create_point_heatmap(coords, image_shape, sigma=10):
    """Create a Gaussian heatmap from point coordinates"""
    x, y = int(coords[0]), int(coords[1])
    height, width = image_shape
    
    if x < 0 or y < 0 or x >= width or y >= height:
        # Point is outside the image, use center
        x, y = width // 2, height // 2
    
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[y, x] = 1
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heatmap = heatmap / heatmap.max()
    
    return heatmap

def predict_segmentation(image, coords, model, device, dim=256):
    """Predict segmentation from image and point coordinates"""
    # Preprocess the image
    img_np = np.array(image)
    orig_shape = img_np.shape[:2]  # Store original shape for resizing back
    img_tensor, _, _ = preprocessing(img_np, np.zeros_like(img_np[:,:,0]), mode='valTest', dim=dim)
    
    # Create heatmap from coordinates
    heatmap = create_point_heatmap(coords, (dim, dim), sigma=10)
    heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)
    
    # Prepare inputs for model
    img_tensor = img_tensor.unsqueeze(0).to(device)
    heatmap_tensor = heatmap_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor, heatmap_tensor)
        # Resize output to original image size
        output = F.interpolate(output, size=orig_shape, mode='bilinear', align_corners=False)
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    # Create color-coded segmentation mask
    mask_colored = np.zeros((orig_shape[0], orig_shape[1], 3), dtype=np.uint8)
    
    # Background: black
    mask_colored[prediction == 0] = [0, 0, 0]
    # Cat: blue
    mask_colored[prediction == 1] = [31, 119, 180]
    # Dog: orange
    mask_colored[prediction == 2] = [255, 127, 14]
    
    # Create an overlay by blending the original image with the mask
    alpha = 0.5
    overlay = cv2.addWeighted(
        img_np, 1-alpha, 
        mask_colored, alpha, 
        0
    )
    
    # Return both the raw prediction and visualization
    return {
        'segmentation': Image.fromarray(mask_colored),
        'overlay': Image.fromarray(overlay)
    }

def click_and_predict(image, evt: gr.SelectData, model, device, dim=256):
    """Handle click event and predict segmentation"""
    if image is None:
        return {
            "segmentation": None,
            "overlay": None
        }
    
    coords = evt.index
    return predict_segmentation(image, coords, model, device, dim)

def create_gradio_interface():
    # Settings
    model_path = "src/models/checkpoints/best_model.pth"  # Update with your actual model path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 256
    
    # Load model
    model = load_model(model_path, device)
    
    # Define UI components
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive Pet Segmentation")
        gr.Markdown("Upload an image and click on a pet (cat or dog) to segment it.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                gr.Markdown("Click on a pet to segment it")
            
            with gr.Column():
                output = gr.Gallery(
                    label="Segmentation Result",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    object_fit="contain"
                )
        
        # Set up the click event
        input_image.select(
            fn=lambda img, evt: click_and_predict(img, evt, model, device, dim),
            inputs=[input_image],
            outputs=output
        )
        
        # Example images
        examples = [
            ["example1.jpg"],
            ["example2.jpg"]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=input_image,
        )
        
        gr.Markdown("## Legend")
        gr.Markdown("- Black: Background")
        gr.Markdown("- Blue: Cat")
        gr.Markdown("- Orange: Dog")
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch() 