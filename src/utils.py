import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_device():
    """Returns the torch.device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[Tenuto] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Tenuto] Using CPU")
    return device

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Tenuto] Random seed set to {seed}")

def mount_google_drive(mount_point: str = "/content/drive"):
    """Mounts Google Drive when executing inside Google Colab."""
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"[Tenuto] Mounted Google Drive at {mount_point}")
    except ImportError:
        print("[Tenuto] Not running in Google Colab environment. Skipped drive mount.")

def save_checkpoint(state, checkpoint_dir: str = "checkpoints", filename: str = "model_best.pth"):
    """Saves model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"[Tenuto] Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None):
    """Loads model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
    checkpoint = torch.load(filepath, map_location=get_device())
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"[Tenuto] Checkpoint loaded from {filepath}")
    return checkpoint
