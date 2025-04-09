import os
os.system("pip show pydantic")
os.system("pip install fastapi==0.112.2 gradio==4.44.1 torch numpy albumentations huggingface_hub")
os.system("pip install git+https://github.com/openai/CLIP.git")
os.system("pip install --upgrade gradio gradio-client")

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from models.unet_model import UNet
from models.autoencoder_model import Autoencoder
from models.clip_model import CLIPUNet, PointCLIPUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Directory to store downloaded weights
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def download_model_weights(repo_id, filename):
    """Download model weights from Hugging Face Hub if they don't exist locally"""
    local_path = os.path.join(WEIGHTS_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"Model weights already exist at {local_path}")
        return local_path
    
    print(f"Downloading model weights from {repo_id}...")
    try:
        # Download the file from Hugging Face
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # If the download is to a different location, copy to our weights directory
        if downloaded_path != local_path:
            print(f"Copying weights from {downloaded_path} to {local_path}")
            os.system(f"cp {downloaded_path} {local_path}")
            
        print(f"Downloaded model weights to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        return None

# Function to preload all model weights during application startup
def preload_all_model_weights():
    """Download all model weights at startup"""
    print("Preloading all model weights...")
    
    for model_name, model_info in MODEL_INFO.items():
        if "huggingface_repo" in model_info and "huggingface_filename" in model_info:
            repo_id = model_info["huggingface_repo"]
            filename = model_info["huggingface_filename"]
            print(f"Preloading weights for {model_name} from {repo_id}")
            downloaded_path = download_model_weights(repo_id, filename)
            
            if downloaded_path:
                print(f"Successfully preloaded weights for {model_name}")
            else:
                print(f"Failed to preload weights for {model_name}")
    
    print("Finished preloading all model weights")

def preprocessing(img: np.ndarray, mask: np.ndarray, mode: str = 'train', dim: int = 256):
    """Preprocess the image and mask for training.
    mode 'train' is for training, applying data augmentation and resizing.
    mode 'valTest' is for validation/testing, applying normalization only.
    
    Args:
        img (np.ndarray): Image as NumPy array.
        mask (np.ndarray): Mask as NumPy array.
        dim (int): Target image dimension.
        mode (str): One of 'train', 'valTest'.
        
    Returns:
        Tuple[torch.Tensor]: Processed image and mask as PyTorch tensors.
    """
    
    assert mode in ['train', 'valTest'], f'Invalid mode: {mode}'
    assert dim > 0, f'Invalid image dimension: {dim}'
    
    # Define common transformations for standard resizing and normalization    
    resizing = A.Compose([
        A.LongestMaxSize(max_size=dim, interpolation=0),
        A.PadIfNeeded(min_height=dim, min_width=dim, border_mode=0)
    ])
    normalisation = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    if mode == 'valTest':
        augmentation = A.Compose([
            resizing,
            normalisation,
            ToTensorV2()
        ])
    else:
        # Define transformations for augmentation
        augmentation = A.Compose([
            resizing,
            
            #### ADD AUGMENTATION HERE ####
            
            # A.RandomCrop(img_dim, img_dim),  # Crop to fixed size
            A.HorizontalFlip(p=0.5),  # Flip images & masks with 50% probability
            A.Rotate(limit=20, p=0.5),  # Random rotation (-20° to 20°)
            # A.ElasticTransform(alpha=1, sigma=50, p=0.1),  # Elastic distortion
            # A.GridDistortion(p=0.3),  # Slight grid warping
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Color jitter
            # A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Random blur
            # A.GaussNoise(var_limit=(10, 50), p=0.2),  # Random noise
            # A.CoarseDropout(max_holes=2, max_height=50, max_width=50, p=0.3),  # Cutout occlusion
            
            ### END AUGMENTATION ###
            
            normalisation, 
            ToTensorV2()  # Convert to PyTorch tensor
        ])
        
    original_mask = torch.tensor(mask, dtype=torch.long)
    augmented = augmentation(image=img, mask=mask)
    return augmented['image'], augmented['mask'], original_mask

# Dictionary of available models and their characteristics
MODEL_INFO = {
    "UNet": {
        "class": UNet,
        "path": "weights/best_unet.pth",
        "huggingface_repo": "bhenriquezsoto/UNet",
        "huggingface_filename": "best_unet.pth",
        "requires_click": False,
        "description": "Standard UNet model for segmentation",
        "available": True
    }
}

# Add models only if they're available
MODEL_INFO.update({
        "ClipUnet": {
            "class": CLIPUNet,
            "path": "weights/clip_unet.pth",
            "huggingface_repo": "bhenriquezsoto/ClipUNet",
            "huggingface_filename": "clip_unet.pth",
            "requires_click": False,
            "description": "UNet with CLIP features for better semantic understanding",
            "available": True
        },
        "ClipUnetPoint": {
            "class": PointCLIPUNet,
            "path": "weights/clip_unet_point_based.pth",
            "huggingface_repo": "bhenriquezsoto/clipUnetPointBased",
            "huggingface_filename": "clip_unet_point_based.pth",
            "requires_click": True,
            "description": "Click-based CLIP UNet for interactive segmentation",
            "available": True
        }
})

MODEL_INFO.update({
    "AutoEncoder": {
        "class": Autoencoder,
        "path": "weights/best_autoencoder.pth",
        "huggingface_repo": "bhenriquezsoto/autoencoder",
        "huggingface_filename": "best_autoencoder.pth",
        "requires_click": False,
        "description": "Autoencoder model for pet segmentation",
        "available": True
        }
    })

# Preload all model weights right after MODEL_INFO is defined
preload_all_model_weights()

# Global variable to store loaded models
loaded_models = {}

def load_model(model_name, device):
    """Load a segmentation model by name"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    print(f"Loading {model_name}...")
    print(MODEL_INFO[model_name])
    
    model_info = MODEL_INFO[model_name]
    model_class = model_info["class"]
    model_path = model_info["path"]
    
    # The weights should already be downloaded during initialization
    # No need to download again, just use the local path
    
    if model_name == "UNet":
        model = model_class(n_channels=3, n_classes=3, bilinear=False)
    elif model_name == "ClipUnet":
        model = model_class(n_channels=3, n_classes=3, bilinear=False)
    elif model_name == "ClipUnetPoint":
        model = model_class(n_channels=3, n_classes=3, bilinear=False)
    elif model_name == "AutoEncoder":
        model = model_class(n_channels=3, n_classes=3)
    
    model.to(device)
    
    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        loaded_models[model_name] = model
        print(f"Successfully loaded {model_name} from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        raise

def create_point_heatmap(coords, image_shape, sigma=10):
    """Create a Gaussian heatmap from point coordinates"""
    x, y = int(coords[0]), int(coords[1])
    height, width = image_shape
    
    if x < 0 or y < 0 or x >= width or y >= height:
        # Point is outside the image, use center
        x, y = width // 2, height // 2
    
    # Create heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[y, x] = 1
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heatmap = heatmap / heatmap.max()
    
    print(f"Heatmap peak at coordinates: ({x}, {y})")
    
    return heatmap

def predict_with_click(image, coords, model_name, status_element, device, dim=256):
    """Predict segmentation from image and point coordinates using a click-based model"""
    if image is None:
        return None, "No image provided"
    
    print("Processing...")
    
    # Preprocess the image
    img_np = np.array(image)
    orig_shape = img_np.shape[:2]  # Store original shape for resizing back
    
    # Convert coordinates from original image size to dim x dim
    scale_h, scale_w = dim / orig_shape[0], dim / orig_shape[1]
    scaled_coords = (int(coords[0] * scale_w), int(coords[1] * scale_h))
    print(f"Original coords: {coords}, Scaled coords: {scaled_coords}")
    
    img_tensor, _, _ = preprocessing(img_np, np.zeros_like(img_np[:,:,0]), mode='valTest', dim=dim)
    
    # Create heatmap from SCALED coordinates
    heatmap = create_point_heatmap(scaled_coords, (dim, dim), sigma=5)
    # Add batch and channel dimensions (unsqueeze twice)
    heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    
    # Prepare inputs for model
    img_tensor = img_tensor.unsqueeze(0).to(device)
    heatmap_tensor = heatmap_tensor.to(device)
    
    # Load model
    model = load_model(model_name, device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor, heatmap_tensor)
        # Resize output to original image size
        output = F.interpolate(output, size=orig_shape, mode='bilinear', align_corners=False)
        
        # Get final prediction
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # Determine which class was clicked on
        clicked_class = prediction[int(coords[1]), int(coords[0])]
        print(f"Clicked on class: {clicked_class} (0=background, 1=cat, 2=dog)")
        
        # Create a binary mask for only the clicked class
        mask = (prediction == clicked_class).astype(np.uint8)
    
    # Define colors for classes
    class_colors = {
        0: [255, 0, 0],     # Red for background
        1: [31, 119, 180],  # Blue for cat
        2: [255, 127, 14]   # Orange for dog
    }
    
    # Get the color for the clicked class
    clicked_color = class_colors[clicked_class]
    
    # Create color-coded segmentation mask based on which class was clicked
    mask_colored = np.zeros((orig_shape[0], orig_shape[1], 3), dtype=np.uint8)
    
    # Set the clicked class to its color
    mask_colored[mask == 1] = clicked_color
    
    # Create an overlay by blending the original image with the mask
    alpha = 0.5
    overlay = cv2.addWeighted(
        img_np, 1-alpha, 
        mask_colored, alpha, 
        0
    )
    
    # Get class name for display
    class_names = ["Background", "Cat", "Dog"]
    clicked_class_name = class_names[clicked_class]
    
    # Create images
    overlay_img = Image.fromarray(overlay)
    
    print("Ready")
    return overlay_img, f"Segmented {clicked_class_name}"

def predict_without_click(image, model_name, status_element, device, dim=256):
    """Predict segmentation from image using a non-click model"""
    if image is None:
        return None, "No image provided"
    
    print("Processing...")
    
    # Preprocess the image
    img_np = np.array(image)
    orig_shape = img_np.shape[:2]  # Store original shape for resizing back
    
    img_tensor, _, _ = preprocessing(img_np, np.zeros_like(img_np[:,:,0]), mode='valTest', dim=dim)
    
    # Prepare inputs for model
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Load model
    model = load_model(model_name, device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        # Resize output to original image size
        output = F.interpolate(output, size=orig_shape, mode='bilinear', align_corners=False)
        
        # Get final prediction
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    # Define colors for classes
    class_colors = {
        0: [255, 0, 0],     # Red for background
        1: [31, 119, 180],  # Blue for cat
        2: [255, 127, 14]   # Orange for dog
    }
    
    # Create RGB segmentation mask
    mask_colored = np.zeros((orig_shape[0], orig_shape[1], 3), dtype=np.uint8)
    
    # Set each class to its color
    for class_idx, color in class_colors.items():
        mask_colored[prediction == class_idx] = color
    
    # Create an overlay by blending the original image with the mask
    alpha = 0.5
    overlay = cv2.addWeighted(
        img_np, 1-alpha, 
        mask_colored, alpha, 
        0
    )
    
    # Create images
    overlay_img = Image.fromarray(overlay)
    
    print("Ready")
    return overlay_img, "Full segmentation completed"

def update_status(status_element, message):
    """Update status message in the UI"""
    # Gradio textbox doesn't have update method, return the new value instead
    return message

def load_model_and_update_status(model_name, device):
    """Load a model and update status"""
    # Load the model
    print(f"Loading {model_name}...")
    model = load_model(model_name, device)
    print(f"{model_name} loaded and ready")
    
    # Return both the status message and whether the model requires click
    return f"{model_name} loaded and ready", MODEL_INFO[model_name]["requires_click"]

def create_gradio_interface():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 256
    
    # Get list of available models
    available_models = [name for name, info in MODEL_INFO.items() if info.get("available", False)]
    default_model = available_models[0] if available_models else None
    
    if not available_models:
        print("Error: No models available!")
        return None
    
    # CSS for improved styling
    custom_css = """
    .model-selection {
        margin-bottom: 15px;
    }
    .info-box {
        padding: 15px;
        background-color: #f5f5f5;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .segmentation-controls {
        margin-top: 15px;
    }
    .result-area {
        min-height: 300px;
    }
    .model-info {
        margin-bottom: 20px;
    }
    .status-area {
        font-weight: bold;
    }
    .header-area {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-title {
        flex-grow: 1;
    }
    .header-controls {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .legend-container {
        margin-top: 10px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    .legend-item {
        display: inline-block;
        margin-right: 15px;
    }
    """
    
    # Track currently selected model - this is a plain Python variable
    current_model_name = default_model
    
    # Create interface
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        # Header area with title and run button
        with gr.Row(elem_classes="header-area"):
            with gr.Column(elem_classes="header-title"):
                gr.Markdown("# Interactive Pet Segmentation")
                gr.Markdown("Upload an image and select a model to segment pets (cats and dogs)")
            
            with gr.Column(elem_classes="header-controls"):
                run_button = gr.Button(
                    "▶️ Run Segmentation", 
                    variant="primary", 
                    size="lg", 
                    visible=True
                )
                status_text = gr.Textbox(
                    label="Status", 
                    value="Select a model", 
                    interactive=False,
                    elem_classes="status-area"
                )
        
        # State variables
        requires_click = gr.State(False)
        
        with gr.Row(equal_height=True):
            # Left column - inputs
            with gr.Column(scale=3):
                with gr.Row(elem_classes="model-selection"):
                    model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=available_models,
                        value=default_model,
                        info="Choose a segmentation model"
                    )
                
                # Input image
                input_image = gr.Image(label="Input Image", type="pil", height=400)
                
                # Model info
                with gr.Row(elem_classes="model-info"):
                    model_info = gr.Markdown("Select a model to see information")
                
                with gr.Row(elem_classes="segmentation-controls"):
                    click_text = gr.Markdown(
                        "👆 **Click on a pet (or background) in the image to segment it**", 
                        visible=False
                    )
            
            # Right column - outputs
            with gr.Column(scale=3):
                output_image = gr.Image(
                    label="Segmentation Result", 
                    elem_classes="result-area",
                    height=400
                )
                output_text = gr.Textbox(
                    label="Result Details", 
                    interactive=False
                )
                
                # Legend moved here - right below the output
                with gr.Row(elem_classes="legend-container"):
                    gr.Markdown("**Legend:** 🔴 Red: Background | 🔵 Blue: Cat | 🟠 Orange: Dog")
        
        # Update UI when model is selected
        def on_model_selected(model_name):
            nonlocal current_model_name
            current_model_name = model_name
            
            if not model_name:
                return (
                    "No model selected", 
                    "Please select a model",
                    gr.update(visible=False),
                    False
                )
                
            info = MODEL_INFO[model_name]
            needs_click = info["requires_click"]
            
            try:
                load_model(model_name, device)
                status = f"{model_name} loaded and ready"
                
                # Update model info text
                model_info_text = f"**{model_name}**\n\n{info['description']}\n\n"
                model_info_text += "⚠️ This model requires you to click on a region to segment." if needs_click else "ℹ️ This model will segment the entire image automatically."
                
                return (
                    model_info_text,
                    status,
                    gr.update(visible=needs_click),
                    needs_click
                )
            except Exception as e:
                error_msg = f"Error loading {model_name}: {str(e)}"
                return (
                    f"**Error Loading Model**\n\n{str(e)}",
                    error_msg,
                    gr.update(visible=False),
                    False
                )
        
        # Model selection handler - now run button is always visible
        model_dropdown.change(
            fn=on_model_selected,
            inputs=[model_dropdown],
            outputs=[model_info, status_text, click_text, requires_click]
        )
        
        # Click event handler
        def on_image_click(image, evt: gr.SelectData):
            if image is None:
                return None, "No image uploaded"
                
            try:
                if not hasattr(evt, 'index'):
                    return None, "Click detection failed"
                    
                coords = evt.index
                print(f"Click event coordinates: {coords}")
                
                # Use the current model name from our variable
                if current_model_name:
                    print(f"Using model: {current_model_name}")
                    # Only use click-based prediction if the model requires it
                    if MODEL_INFO[current_model_name]["requires_click"]:
                        result_img, result_text = predict_with_click(image, coords, current_model_name, None, device, dim)
                        return result_img, result_text
                    else:
                        # For non-click models, we'll ignore the click and use the run button
                        return None, "This model doesn't use clicks. Please use the Run Segmentation button."
                else:
                    return None, "Please select a model first"
            except Exception as e:
                print(f"Error in image click: {e}")
                import traceback
                traceback.print_exc()
                return None, f"Error: {str(e)}"
        
        # Register click handler
        input_image.select(
            fn=on_image_click,
            inputs=[input_image],
            outputs=[output_image, output_text]
        )
        
        # Non-click model handler - now a common handler that checks if click is required
        def on_run_button_click(image):
            if image is None:
                return None, "No image uploaded"
            
            try:
                # Use the global current model
                if current_model_name:
                    print(f"Running segmentation with {current_model_name}")
                    # Check if model requires click
                    if MODEL_INFO[current_model_name]["requires_click"]:
                        return None, "This model requires you to click on the image to segment"
                    else:
                        result_img, result_text = predict_without_click(image, current_model_name, None, device, dim)
                        return result_img, result_text
                else:
                    return None, "Please select a model first"
            except Exception as e:
                print(f"Error in segmentation: {e}")
                import traceback
                traceback.print_exc()
                return None, f"Error: {str(e)}"
        
        # Register run button handler - always visible now
        run_button.click(
            fn=on_run_button_click,
            inputs=[input_image],
            outputs=[output_image, output_text]
        )
        
        # Load default model on startup
        if default_model:
            demo.load(
                fn=lambda: on_model_selected(default_model),
                inputs=None,
                outputs=[model_info, status_text, click_text, requires_click]
            )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch() 