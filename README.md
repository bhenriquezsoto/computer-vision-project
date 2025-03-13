# Computer Vision Project: Pet Segmentation

🎮 **Try our interactive demo!** Upload an image and click on a pet to generate its segmentation mask:  
➡️ [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/bhenriquezsoto/point-based-segmentation-1)

## About
This repository contains the 2024/25 Computer Vision course project at the University of Edinburgh, focusing on pet segmentation using various deep learning approaches.

## Authors
- [Julia Lopez Gomez](https://github.com/julialopezgomez)
- [Benjamin Henriquez Soto](https://github.com/bhenriquezsoto)

## Model Performance Comparison

### Point-based UNet
Using our performing model (**UNet**) with interactive point-based segmentation.

#### Results
- **Overall Metrics**
  - Mean Dice Score: 0.7975
  - Mean IoU: 0.6859 
  - Pixel Accuracy: 0.8734

- **Per-Class Performance**
  - Background: Dice 0.9074, IoU 0.8388
  - Cat: Dice 0.7337, IoU 0.6048
  - Dog: Dice 0.7513, IoU 0.6141

- **Training Parameters**
  - Batch size: 32
  - Epochs: 70
  - Learning rate: 0.001
  - Class weights: Background 0.3, Cat 2.2, Dog 0.8

To train: `python src/train.py --model point_unet --classes 3 --epochs 70 --batch-size 32 --learning-rate 0.0001 --img-dim 256 --class-weights 0.3 2.2 0.8 --amp`

### Standard UNet
#### Results
- **Overall Metrics**
  - Mean Dice Score: 0.7475
  - Mean IoU: 0.6805
  - Pixel Accuracy: 0.8740

- **Per-Class Performance**
  - Background: Dice 0.9083, IoU 0.8403
  - Cat: Dice 0.7200, IoU 0.6779
  - Dog: Dice 0.6142, IoU 0.5234

### CLIP-based Approach
#### Results
- **Overall Metrics**
  - Mean Dice Score: 0.6118
  - Mean IoU: 0.5738
  - Pixel Accuracy: 0.7169

- **Per-Class Performance**
  - Background: Dice 0.7792, IoU 0.6754
  - Cat: Dice 0.6920, IoU 0.6889
  - Dog: Dice 0.3642, IoU 0.3571

### Auto-Encoder Approach
#### Results
- **Overall Metrics**
  - Mean Dice Score: 0.6317
  - Mean IoU: 0.5631
  - Pixel Accuracy: 0.7370

- **Per-Class Performance**
  - Background: Dice 0.8469, IoU 0.7468
  - Cat: Dice 0.6730, IoU 0.6730
  - Dog: Dice 0.3754, IoU 0.2696

- **Training Parameters**
  - Batch size: 128
  - Epochs: 100 (50 reconstruction + 50 segmentation)
  - Learning rate: 0.001

To train: `python src/train.py -e 100 --model autoencoder --amp --batch-size 128`
