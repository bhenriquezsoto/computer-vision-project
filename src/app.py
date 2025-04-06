import gradio as gr
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2

from models.unet_model import PointUNet
from data_loading import preprocessing

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
    
    # Create heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[y, x] = 1
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heatmap = heatmap / heatmap.max()
    
    print(f"Heatmap peak at coordinates: ({x}, {y})")
    
    return heatmap

def predict_segmentation(image, coords, model, device, dim=256):
    """Predict segmentation from image and point coordinates"""
    # Preprocess the image
    img_np = np.array(image)
    orig_shape = img_np.shape[:2]  # Store original shape for resizing back
    
    # Convert coordinates from original image size to dim x dim
    scale_h, scale_w = dim / orig_shape[0], dim / orig_shape[1]
    scaled_coords = (int(coords[0] * scale_w), int(coords[1] * scale_h))
    print(f"Original coords: {coords}, Scaled coords: {scaled_coords}")
    
    img_tensor, _, _ = preprocessing(img_np, np.zeros_like(img_np[:,:,0]), mode='valTest', dim=dim)
    
    # Create heatmap from SCALED coordinates
    heatmap = create_point_heatmap(scaled_coords, (dim, dim), sigma=5)  # Reduced sigma for more precise pointing
    # Add batch and channel dimensions (unsqueeze twice)
    heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    
    # Prepare inputs for model
    img_tensor = img_tensor.unsqueeze(0).to(device)
    heatmap_tensor = heatmap_tensor.to(device)
    
    # Print shapes for debugging
    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Heatmap tensor shape: {heatmap_tensor.shape}")
    
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
    mask_img = Image.fromarray(mask_colored)
    
    return [
        (overlay_img, f"Overlay - {clicked_class_name}"), 
        (mask_img, f"Mask - {clicked_class_name}")
    ]

def click_and_predict(image, evt: gr.SelectData, model, device, dim=256):
    """Handle click event and predict segmentation"""
    if image is None:
        return None
    
    try:
        # Get the coordinates from the click event
        if hasattr(evt, 'index'):
            coords = evt.index
            print(f"Using click coordinates: {coords}")
        else:
            # If no click coordinates, use center of image as default
            print("No click coordinates found, using center of image")
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            coords = (w//2, h//2)
            print(f"Using default coordinates: {coords}")
        
        result = predict_segmentation(image, coords, model, device, dim)
        print("Segmentation completed successfully")
        return result
    except Exception as e:
        print(f"Error in click_and_predict: {e}")
        import traceback
        traceback.print_exc()
        # Return empty result on error
        return None

def create_gradio_interface():
    # Settings
    model_path = "weights/best_model_after_epoch_10_swept-fire-9_weighted.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 256
    
    # Load model
    model = load_model(model_path, device)
    
    # Define UI components
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive Pet Segmentation")
        gr.Markdown("Upload an image and click on a pet (cat or dog) or background to segment it.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                gr.Markdown("Click on a pet (or background) to segment it")
            
            with gr.Column():
                output = gr.Gallery(
                    label="Segmentation Result",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height="auto",
                    object_fit="contain"
                )
        
        # Set up the click event with a completely different approach
        def on_image_select(image, evt: gr.SelectData):
            print(f"Event received: {evt}")
            print(f"Image: {type(image)}")
            print(f"Event type: {type(evt)}")
            if hasattr(evt, 'index'):
                print(f"Click coordinates: {evt.index}")
            return click_and_predict(image, evt, model, device, dim)
        
        # Clear the output before setting up the event handler
        input_image.clear(outputs=output)
        
        # Register the click event
        input_image.select(
            fn=on_image_select,
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
        gr.Markdown("- Red: Background")
        gr.Markdown("- Blue: Cat")
        gr.Markdown("- Orange: Dog")
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch() 