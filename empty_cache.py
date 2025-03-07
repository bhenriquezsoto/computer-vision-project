import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU cache cleared.")
else:
    print("No GPU available to clear cache.")