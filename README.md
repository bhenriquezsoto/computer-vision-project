# computer-vision-project

This repository contains the 2024/25 project version of the Computer Vision course at the University of Edinburgh

## Authors
- [Julia Lopez Gomez](https://github.com/julialopezgomez)
- [Benjamin Henriquez Soto](https://github.com/bhenriquezsoto)

## Results

### UNet

### CLIP

### Auto-Encoder Approach
In here we obtain the following results in the test set:

#### General results

- Mean Dice Score: 0.6317                                                               
- Mean IoU: 0.5631
- Pixel Accuracy: 0.7370

#### Results by class
- Class 0 - Dice: 0.8469, IoU: 0.7468
- Class 1 - Dice: 0.6730, IoU: 0.6730
- Class 2 - Dice: 0.3754, IoU: 0.2696


The parameters were the following:

- Batch size: 128
- Epochs: 100 (50 for the reconstruction phase and 50 for the segmentation phase)
- Initial learning-rate: 0.001

Can be tested with `python ./src/train.py -e 100 --model autoencoder --amp --batch-size 128`

### Prompt-based Segmentation
