import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TenutoScoreDataset(Dataset):
    """
    PyTorch Dataset for Note Sequences extracted from MusicXML / score files and aligned performances.
    Supports both 40D Note Feature Matrix (Full Pipeline) and 10D (Prototypes).
    """
    def __init__(self, data_dir: str, seq_len: int = 256, in_features: int = 40, is_synthetic: bool = False):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.in_features = in_features
        self.samples = []
        self._file_cache = {}
        self._max_cache_size = 500

        # Auto-detect preprocessed .pt files in data_dir
        raw_files = []
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.pt') or f.endswith('.npz'):
                        raw_files.append(os.path.join(root, f))
        
        # If no .pt files found, raise error
        if len(raw_files) == 0:
            raise FileNotFoundError(f"[TenutoDataset] Error: No preprocessed dataset files (.pt) found in '{data_dir}'. Please run dataset preprocessing first.")
        else:
            # Chunk the raw files into seq_len-note sequences with 50% overlap (hop = seq_len // 2)
            hop_size = seq_len // 2
            for filepath in raw_files:
                try:
                    data = torch.load(filepath, map_location="cpu")
                    x = data.get("x")
                    if x is None:
                        continue
                    if x.ndim == 1:
                        x = x.unsqueeze(0)
                    num_notes = x.size(0)
                    if num_notes == 0:
                        continue
                    if num_notes <= seq_len:
                        self.samples.append({"path": filepath, "start_idx": 0})
                    else:
                        for start_idx in range(0, num_notes - hop_size, hop_size):
                            self.samples.append({"path": filepath, "start_idx": start_idx})
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
            print(f"[TenutoDataset] Successfully loaded {len(raw_files)} files into {len(self.samples)} chunks from '{data_dir}'.")

    def __len__(self):
        if hasattr(self, 'synthetic_num_samples'):
            return self.synthetic_num_samples
        return len(self.samples)

    def _get_data(self, filepath):
        if filepath in self._file_cache:
            return self._file_cache[filepath]
        data = torch.load(filepath, map_location="cpu")
        if len(self._file_cache) < self._max_cache_size:
            self._file_cache[filepath] = data
        return data

    def __getitem__(self, idx):
        if hasattr(self, 'synthetic_num_samples'):
            # Synthetic 40D or 10D Note Feature Sequence X in R^(SeqLen x InFeatures)
            x = torch.randn(self.seq_len, self.in_features)
            
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
            sample_info = self.samples[idx]
            filepath = sample_info["path"]
            start_idx = sample_info["start_idx"]
            
            data = self._get_data(filepath)
            x = data["x"]
            
            # Ensure x is 2D (SeqLen, InFeatures)
            if x.ndim == 1:
                x = x.unsqueeze(0)

            # Extract the chunk
            end_idx = min(start_idx + self.seq_len, x.size(0))
            x_chunk = x[start_idx:end_idx]

            # Pad or truncate x to fixed seq_len if needed
            if x_chunk.size(0) < self.seq_len:
                pad_size = self.seq_len - x_chunk.size(0)
                pad = torch.zeros(pad_size, x_chunk.size(1))
                x_chunk = torch.cat([x_chunk, pad], dim=0)

            targets = data.get("targets", {})
            if not targets:
                from src.alignment import compute_alignment_targets
                targets = compute_alignment_targets(None, None)
            
            targets_chunk = {}
            for k in ["delta_t", "velocity", "articulation", "pedal", "tempo_scale"]:
                if k in targets:
                    v = targets[k]
                    v_chunk = v[start_idx:end_idx] if start_idx < v.size(0) else torch.zeros(0)
                    if v_chunk.size(0) < self.seq_len:
                        pad_v = torch.zeros(self.seq_len - v_chunk.size(0))
                        targets_chunk[k] = torch.cat([v_chunk, pad_v], dim=0)
                    else:
                        targets_chunk[k] = v_chunk[:self.seq_len]

            return x_chunk, targets_chunk

def create_dataloaders(data_dir: str, batch_size: int = 16, seq_len: int = 256, in_features: int = 40, num_workers: int = 2):
    """Creates train and validation PyTorch DataLoaders for score sequences."""
    train_dataset = TenutoScoreDataset(os.path.join(data_dir, "train"), seq_len=seq_len, in_features=in_features)
    val_dataset = TenutoScoreDataset(os.path.join(data_dir, "val"), seq_len=seq_len, in_features=in_features)

    nw = num_workers if (os.cpu_count() and os.cpu_count() > 1) else 0
    import torch.multiprocessing as mp
    ctx = mp.get_context("forkserver") if (nw > 0 and "forkserver" in mp.get_all_start_methods()) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True if nw > 0 else False, multiprocessing_context=ctx)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True if nw > 0 else False, multiprocessing_context=ctx)

    return train_loader, val_loader
