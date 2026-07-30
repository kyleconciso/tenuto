import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TenutoScoreDataset(Dataset):
    """
    PyTorch Dataset for Note Sequences extracted from MusicXML / score files and aligned performances.
    Supports both 40D Note Feature Matrix (Full Pipeline) and 10D (Prototypes).
    
    Sequences are formatted as fixed-length chunks (e.g., 256 notes per sequence).
    """
    def __init__(self, data_dir: str, seq_len: int = 256, in_features: int = 40, is_synthetic: bool = True):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.in_features = in_features
        self.samples = []

        # If data_dir contains saved numpy/pt files, load them
        if os.path.exists(data_dir) and not is_synthetic:
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.pt') or f.endswith('.npz'):
                        self.samples.append(os.path.join(root, f))
        
        # If synthetic or no files found, generate synthetic score sequence samples for testing pipeline
        if len(self.samples) == 0:
            print(f"[TenutoDataset] No pre-processed datasets found in '{data_dir}'. Generating synthetic note sequences for pipeline verification...")
            self.synthetic_num_samples = 100

    def __len__(self):
        if hasattr(self, 'synthetic_num_samples'):
            return self.synthetic_num_samples
        return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self, 'synthetic_num_samples'):
            # Synthetic 40D or 10D Note Feature Sequence X in R^(SeqLen x InFeatures)
            x = torch.randn(self.seq_len, self.in_features)
            
            # Ground-Truth Targets Y: \Delta t, v, d, pedal
            delta_t = torch.clamp(0.01 * torch.randn(self.seq_len), min=-0.025, max=0.025)
            velocity = torch.clamp(64.0 + 15.0 * torch.randn(self.seq_len), min=0.0, max=127.0)
            articulation = torch.clamp(1.0 + 0.2 * torch.randn(self.seq_len), min=0.1, max=3.0)
            pedal = torch.clamp(64.0 + 30.0 * torch.randn(self.seq_len), min=0.0, max=127.0)
            tempo_scale = torch.clamp(1.0 + 0.1 * torch.randn(self.seq_len), min=0.5, max=1.5)

            targets = {
                "delta_t": delta_t,
                "velocity": velocity,
                "articulation": articulation,
                "pedal": pedal,
                "tempo_scale": tempo_scale
            }
            return x, targets
        else:
            filepath = self.samples[idx]
            data = torch.load(filepath)
            return data["x"], data["targets"]

def extract_partitura_features(xml_path: str):
    """
    Extracts 40D note feature matrix using partitura.musicanalysis.
    """
    try:
        import partitura as pt
        score = pt.load_score(xml_path)
        features = pt.musicanalysis.make_note_features(score)
        return torch.tensor(features, dtype=torch.float32)
    except ImportError:
        print("[TenutoDataset] partitura package not installed. Install via `pip install partitura`.")
        return None

def create_dataloaders(data_dir: str, batch_size: int = 16, seq_len: int = 256, in_features: int = 40, num_workers: int = 0):
    """
    Creates train and validation PyTorch DataLoaders for score sequences.
    """
    train_dataset = TenutoScoreDataset(os.path.join(data_dir, "train"), seq_len=seq_len, in_features=in_features)
    val_dataset = TenutoScoreDataset(os.path.join(data_dir, "val"), seq_len=seq_len, in_features=in_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
