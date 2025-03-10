#TODO: Implement testing function

def test_model(net, dataloather, device, amp, n_classes=3, desc='Testing round'):
    """
    Tests model on the testing dataset.

    Returns:
    - Mean Dice Score
    - Mean IoU
    - Mean Pixel Accuracy
    - Per-class Dice Scores
    - Per-class IoU Scores
    """