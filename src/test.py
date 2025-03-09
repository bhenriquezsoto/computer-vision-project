from evaluate import evaluate

def test_model(net, dataloader, device, amp, n_classes=3, desc='Testing round'):
    """
    Tests model on the testing dataset.

    Returns:
    - Mean Dice Score
    - Mean IoU
    - Mean Pixel Accuracy
    - Per-class Dice Scores
    - Per-class IoU Scores
    """
    # If model is an Autoencoder, make sure we're in segmentation phase
    if hasattr(net, 'set_phase'):
        net.set_phase("segmentation")
        
    # Use the evaluate function to test the model
    return evaluate(net, dataloader, device, amp, n_classes=n_classes, desc=desc)