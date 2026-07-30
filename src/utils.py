import os
import random
import numpy as np
import torch

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
        drive.mount(mount_point, force_remount=False)
        print(f"[Tenuto] Mounted Google Drive at {mount_point}")
        return True
    except Exception:
        print("[Tenuto] Not running in Google Colab environment or Drive mount skipped.")
        return False

def get_checkpoint_dir():
    """Returns Google Drive checkpoint path if mounted, else local directory."""
    gdrive_dir = "/content/drive/MyDrive/tenuto_checkpoints"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(gdrive_dir, exist_ok=True)
        return gdrive_dir
    local_dir = "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    return local_dir

def save_checkpoint(state, checkpoint_dir: str = None, filename: str = "best_transformer_model.pth"):
    """Saves checkpoint to both local checkpoints/ and Google Drive if available."""
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    primary_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, primary_path)
    print(f"[Tenuto] Checkpoint saved to '{primary_path}'")

    # Also save to local fallback if primary path is GDrive
    if checkpoint_dir.startswith("/content/drive"):
        local_dir = "checkpoints"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        torch.save(state, local_path)

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None):
    """
    Idempotent checkpoint loader: Tries specified path, GDrive path, or local checkpoints/ directory.
    """
    candidates = [
        filepath,
        os.path.join(get_checkpoint_dir(), os.path.basename(filepath)),
        os.path.join("checkpoints", os.path.basename(filepath))
    ]

    loaded_path = None
    for path in candidates:
        if path and os.path.exists(path):
            loaded_path = path
            break

    if not loaded_path:
        print(f"[Tenuto] No checkpoint found at candidate paths. Starting from scratch.")
        return None

    checkpoint = torch.load(loaded_path, map_location=get_device())
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model.load_state_dict(checkpoint)

    print(f"[Tenuto] Successfully loaded checkpoint from '{loaded_path}'")
    return checkpoint
