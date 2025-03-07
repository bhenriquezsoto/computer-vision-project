import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPSegmentationModel(nn.Module):
    def __init__(self, n_classes):
        super(CLIPSegmentationModel, self).__init__()
        self.n_classes = n_classes
        
        # Load the pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Add a segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Extract features using CLIP's vision encoder
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values=x)
        
        # Reshape the features to match the expected input for the segmentation head
        clip_features = clip_outputs.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 512, 1, 1]
        clip_features = F.interpolate(clip_features, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Pass through the segmentation head
        logits = self.segmentation_head(clip_features)
        return logits