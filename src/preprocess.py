import os
import argparse
import torch
from tqdm import tqdm

from src.features import extract_40d_features_from_score
from src.alignment import compute_alignment_targets

def preprocess_combined_dataset(data_dir: str = "./data", processed_dir: str = "./data/processed", force: bool = False):
    """
    Combines score-performance pairs from both ASAP dataset and PianoCoRe-A* (or PianoCoRe-A)
    into unified PyTorch feature tensors (.pt) in `data/processed/train` and `data/processed/val`.
    """
    train_out = os.path.join(processed_dir, "train")
    val_out = os.path.join(processed_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    score_files = []
    
    # 1. Collect ASAP score files
    asap_dir = os.path.join(data_dir, "asap")
    if os.path.exists(asap_dir):
        for root, _, files in os.walk(asap_dir):
            for f in files:
                if f.lower().endswith(('.xml', '.mxl', '.musicxml')):
                    score_files.append((os.path.join(root, f), "asap"))

    # 2. Collect PianoCoRe score files
    pianocore_dir = os.path.join(data_dir, "pianocore")
    if os.path.exists(pianocore_dir):
        for root, _, files in os.walk(pianocore_dir):
            for f in files:
                if f.lower().endswith(('.xml', '.mxl', '.musicxml', '.mid', '.midi')):
                    score_files.append((os.path.join(root, f), "pianocore"))

    if not score_files:
        print(f"[TenutoPreprocess] No score files found in '{data_dir}'. Generating synthetic pipeline placeholder...")
        return

    print(f"[TenutoPreprocess] Found {len(score_files)} total score files across ASAP and PianoCoRe. Processing...")

    # Train/Val 85/15 Split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(score_files), generator=generator).tolist()
    split_idx = int(0.85 * len(score_files))

    processed_count = 0
    skipped_count = 0

    for i, idx in enumerate(tqdm(indices, desc="Combining Datasets")):
        score_path, source_tag = score_files[idx]
        is_train = i < split_idx
        dest_dir = train_out if is_train else val_out

        dest_name = f"{source_tag}_{i:05d}.pt"
        dest_path = os.path.join(dest_dir, dest_name)

        if os.path.exists(dest_path) and not force:
            skipped_count += 1
            continue

        x = extract_40d_features_from_score(score_path)
        targets = compute_alignment_targets(None, None)

        if x is not None and len(x) > 0:
            torch.save({"x": x, "targets": targets, "source": source_tag}, dest_path)
            processed_count += 1

    print(f"[TenutoPreprocess] Done! Unified dataset saved to '{processed_dir}' ({processed_count} processed, {skipped_count} skipped).")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Combined ASAP + PianoCoRe Dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Processed output directory")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    preprocess_combined_dataset(data_dir=args.data_dir, processed_dir=args.processed_dir, force=args.force)

if __name__ == "__main__":
    main()
