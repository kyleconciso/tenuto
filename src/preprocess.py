import os
import argparse
import torch
from tqdm import tqdm

from src.features import extract_40d_features_from_score
from src.alignment import compute_alignment_targets

def preprocess_asap_dataset(asap_dir: str = "./data/asap", processed_dir: str = "./data/processed", force: bool = False):
    """
    Scans ASAP dataset directory, extracts 40D features from score files,
    derives targets, and saves PyTorch .pt files into `data/processed/train` and `data/processed/val`.
    """
    train_out = os.path.join(processed_dir, "train")
    val_out = os.path.join(processed_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    if not os.path.exists(asap_dir):
        print(f"[TenutoPreprocess] ASAP directory '{asap_dir}' not found. Run `python -m src.download_dataset` first.")
        return

    score_files = []
    for root, _, files in os.walk(asap_dir):
        for f in files:
            if f.lower().endswith(('.xml', '.mxl', '.musicxml')) or (f.lower().endswith('.mid') and 'score' in f.lower()):
                score_files.append(os.path.join(root, f))

    if not score_files:
        print(f"[TenutoPreprocess] No score files found in '{asap_dir}'.")
        return

    print(f"[TenutoPreprocess] Found {len(score_files)} score files in ASAP dataset. Processing into 40D tensors...")
    
    # 80/20 Train/Val Split
    np_rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(score_files), generator=np_rng).tolist()
    split_idx = int(0.8 * len(score_files))

    processed_count = 0
    skipped_count = 0

    for i, idx in enumerate(tqdm(indices, desc="Processing ASAP")):
        score_path = score_files[idx]
        is_train = i < split_idx
        dest_dir = train_out if is_train else val_out
        
        rel_name = f"asap_{i:04d}.pt"
        dest_path = os.path.join(dest_dir, rel_name)

        if os.path.exists(dest_path) and not force:
            skipped_count += 1
            continue

        x = extract_40d_features_from_score(score_path)
        targets = compute_alignment_targets(None, None)

        if x is not None and len(x) > 0:
            torch.save({"x": x, "targets": targets}, dest_path)
            processed_count += 1

    print(f"[TenutoPreprocess] Completed! Saved {processed_count} files ({skipped_count} skipped).")

def main():
    parser = argparse.ArgumentParser(description="Preprocess ASAP Dataset for Tenuto")
    parser.add_argument("--asap_dir", type=str, default="./data/asap", help="Path to ASAP dataset root")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    preprocess_asap_dataset(asap_dir=args.asap_dir, processed_dir=args.processed_dir, force=args.force)

if __name__ == "__main__":
    main()
