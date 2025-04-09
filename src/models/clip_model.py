import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# class CLIPSegmentationModelBad(nn.Module):
#     def __init__(self, n_classes):
#         super(CLIPSegmentationModel, self).__init__()
#         self.n_classes = n_classes
        
#         # Load the pre-trained CLIP model
#         self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
#         # Freeze the CLIP model parameters
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
        
#         # Add a segmentation head
#         self.segmentation_head = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, n_classes, kernel_size=1)
#         )
    
#     def forward(self, x):
#         # Extract features using CLIP's vision encoder
#         with torch.no_grad():
#             clip_outputs = self.clip_model.get_image_features(pixel_values=x)
        
#         # Reshape the features to match the expected input for the segmentation head
#         clip_features = clip_outputs.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 512, 1, 1]
#         clip_features = F.interpolate(clip_features, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
#         # Pass through the segmentation head
#         logits = self.segmentation_head(clip_features)
#         return logits
    
    
class CLIPSegmentationModel(nn.Module):
    def __init__(self, n_classes=3, image_size=256, bilinear=True, dropout_rate=0.0):
        """
        CLIP-only segmentation model. Uses CLIP's image encoder,
        projects the embedding to a spatial feature map, and
        decodes it to a segmentation mask.
        """
        super(CLIPSegmentationModel, self).__init__()
        
        self.n_classes = n_classes
        self.n_channels = 3
        self.image_size = image_size
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()  # freeze CLIP

        bottleneck_channels = 512 if bilinear else 1024
        self.bottleneck_shape = (bottleneck_channels, image_size // 16, image_size // 16)
        C, H, W = self.bottleneck_shape

        self.projector = nn.Linear(512, C * H * W)
        self.decoder = UNetDecoder(
            n_classes=n_classes,
            bilinear=bilinear,
            use_skips=False,
            dropout_rate=dropout_rate
        )

    def forward(self, image):
        """
        Args:
            image: Tensor [B, 3, H, W]
        Returns:
            Segmentation logits: [B, num_classes, H, W]
        """
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            clip_feat = self.clip_model.encode_image(image)  # [B, 512]

        projected = self.projector(clip_feat)  # [B, C * H * W]
        B, C, H, W = image.shape[0], *self.bottleneck_shape
        x = projected.view(B, C, H, W)  # [B, C, H, W]
        print("billinear", self.bilinear)

        return self.decoder(x)
    

class CLIPUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, image_size=256, bilinear=True, use_skips=True, dropout_rate=0.0, fuse_clip=True):
        super(CLIPUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.image_size = image_size
        self.bilinear = bilinear
        self.use_skips = use_skips
        self.dropout_rate = dropout_rate
        
        self.fuse_clip = fuse_clip
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()  # freeze CLIP
        
        bottleneck_channels = 512 if bilinear else 1024
        self.bottleneck_shape = (bottleneck_channels, image_size // 16, image_size // 16)
        C, H, W = self.bottleneck_shape
        self.projector = nn.Linear(512, C * H * W)
        
        self.encoder = UNetEncoder(
            in_channels=n_channels, 
            n_classes=n_classes, 
            bilinear=bilinear, 
            dropout_rate=dropout_rate)
        
        self.decoder = UNetDecoder(
            n_classes=n_classes, 
            bilinear=bilinear, 
            use_skips=use_skips, 
            dropout_rate=dropout_rate)
        
        
        
    def forward(self, x):
        skips, encoder_out = self.encoder(x)
        
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        
        with torch.no_grad():
            clip_feat = self.clip_model.encode_image(x)
        projected = self.projector(clip_feat)  # [B, C * H * W]
        B, C, H, W = x.shape[0], *self.bottleneck_shape
        clip_out = projected.view(B, C, H, W)  # [B, C, H, W]
        
        assert encoder_out.shape == clip_out.shape, f"Encoder output shape {encoder_out.shape} does not match CLIP output shape {clip_out.shape}"
        
        # x = torch.cat([encoder_out, clip_out], dim=1)  # Concatenate encoder output and CLIP output
        if self.fuse_clip:
            x = encoder_out + clip_out  # Add encoder output and CLIP output
        else:
            x = clip_out
        
        return self.decoder(x, skips)
        
        
    
        
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
        
        if x2 != None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            # add padding on x1 to match x2
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            
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
    def __init__(self, n_classes, bilinear=True, use_skips=True, dropout_rate=0.0):
        super(UNetDecoder, self).__init__()
        
        factor = 2 if bilinear else 1
        skip_factor = 2 if ((not use_skips) and bilinear) else 1
        
        # self.up1 = (Up(1024 // skip_factor, 512 // factor, bilinear, dropout_rate=dropout_rate))
        # self.up2 = (Up(512 // skip_factor, 256 // factor, bilinear, dropout_rate=dropout_rate))
        # self.up3 = (Up(256 // skip_factor, 128 // factor, bilinear, dropout_rate=dropout_rate))
        # self.up4 = (Up(128 // skip_factor, 64, bilinear, dropout_rate=dropout_rate))
        # self.outc = (OutConv(64, n_classes))
        print("factor", factor)
        if use_skips:
            print("here")
            self.up1 = (Up(1024, 512 // factor, bilinear, dropout_rate=dropout_rate))
            self.up2 = (Up(512, 256 // factor, bilinear, dropout_rate=dropout_rate))
            self.up3 = (Up(256, 128 // factor, bilinear, dropout_rate=dropout_rate))
            self.up4 = (Up(128, 64, bilinear, dropout_rate=dropout_rate))
            
        else:
            print("here")
            self.up1 = (Up(1024 // factor, 512 // factor, bilinear, dropout_rate=dropout_rate))
            self.up2 = (Up(512 // factor, 256 // factor, bilinear, dropout_rate=dropout_rate))
            self.up3 = (Up(256 // factor, 128 // factor, bilinear, dropout_rate=dropout_rate))
            self.up4 = (Up(128 // factor, 64, bilinear, dropout_rate=dropout_rate))
            
        self.outc = (OutConv(64, n_classes))
        
        
    def forward(self, x, skips=None):
        if skips is None:
            skips = [None, None, None, None]
        x = self.up1(x, skips[3])
        x = self.up2(x, skips[2])
        x = self.up3(x, skips[1])
        x = self.up4(x, skips[0])
        return self.outc(x)
    
    
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, bilinear=True, dropout_rate=0.0):
        super(UNetEncoder, self).__init__()
        
        self.inc = (DoubleConv(in_channels, 64, dropout_rate=dropout_rate))
        self.down1 = (Down(64, 128, dropout_rate=dropout_rate))
        self.down2 = (Down(128, 256, dropout_rate=dropout_rate))
        self.down3 = (Down(256, 512, dropout_rate=dropout_rate))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, dropout_rate=dropout_rate))
        
    def forward(self, x):
        skips = []
        x1 = self.inc(x)
        skips.append(x1)
        x2 = self.down1(x1)
        skips.append(x2)
        x3 = self.down2(x2)
        skips.append(x3)
        x4 = self.down3(x3)
        skips.append(x4)
        x5 = self.down4(x4)
        
        return skips, x5
        
