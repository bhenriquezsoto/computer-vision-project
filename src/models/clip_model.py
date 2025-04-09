import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPSegmentationModelBad(nn.Module):
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
class CLIPSegmentationModel(nn.Module):
    def __init__(self, clip_model, image_size=256, embed_dim=512, num_classes=3, bilinear=True):
        """
        CLIP-only segmentation model. Uses CLIP's image encoder,
        projects the embedding to a spatial feature map, and
        decodes it to a segmentation mask.
        """
        super(CLIPSegmentationModel, self).__init__()

        self.clip_model = clip_model
        self.clip_model.eval()  # freeze CLIP

        bottleneck_channels = 512 if bilinear else 1024
        self.bottleneck_shape = (bottleneck_channels, image_size // 16, image_size // 16)
        C, H, W = self.bottleneck_shape

        self.projector = nn.Linear(bottleneck_channels, C * H * W)
        self.decoder = UNetDecoder(
            in_channels=C,
            skip_channels=[0, 0, 0, 0],  # No skips
            n_classes=num_classes,
            bilinear=bilinear
        )

    def forward(self, image):
        """
        Args:
            image: Tensor [B, 3, H, W]
        Returns:
            Segmentation logits: [B, num_classes, H, W]
        """
        with torch.no_grad():
            clip_feat = self.clip_model.encode_image(image)  # [B, 512]

        projected = self.projector(clip_feat)  # [B, C * H * W]
        B, C, H, W = image.shape[0], *self.bottleneck_shape
        x = projected.view(B, C, H, W)  # [B, C, H, W]

        # Provide dummy skips (all zeros)
        dummy_skips = [torch.zeros((B, 0, H * 2 ** i, W * 2 ** i), device=image.device) for i in reversed(range(4))]

        return self.decoder(x, dummy_skips)
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ]
        
        # Add dropout after the first conv block if dropout_rate > 0
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
            
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Add dropout after the second conv block if dropout_rate > 0
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
            
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # add padding on x1 to match x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 != None:
            x = torch.cat([x2, x1], dim=1) # concatenate along the channel dimension (skip connection)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
        
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True, dropout_rate=0.0):
        super(UNetDecoder, self).__init__()
        
        factor = 2 if bilinear else 1

        self.up1 = Up(in_channels, 1024 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = (Up(512, 256 // factor, bilinear, dropout_rate=dropout_rate))
        self.up3 = (Up(256, 128 // factor, bilinear, dropout_rate=dropout_rate))
        self.up4 = (Up(128, 64, bilinear, dropout_rate=dropout_rate))
        self.outc = (OutConv(64, n_classes))
        
        
    def forward(self, x, skips=[None, None, None, None]):
        x = self.up1(x, skips[3])
        x = self.up2(x, skips[2])
        x = self.up3(x, skips[1])
        x = self.up4(x, skips[0])
        return self.outc(x)
    
    
