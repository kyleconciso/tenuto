import os
import argparse
import torch
from tqdm import tqdm
from src.dataset import extract_partitura_features

def preprocess_dataset(raw_dir: str, processed_dir: str, force: bool = False):
    """
    Idempotently preprocesses score files into PyTorch feature tensors (.pt).
    Skipping files that have already been converted unless `force=True`.
    """
    os.makedirs(processed_dir, exist_ok=True)
    if not os.path.exists(raw_dir):
        print(f"[Tenuto Preprocess] Raw data directory '{raw_dir}' does not exist. Creating directory...")
        os.makedirs(raw_dir, exist_ok=True)
        print(f"[Tenuto Preprocess] Place your MusicXML (.xml/.mxl) or MIDI files into '{raw_dir}'.")
        return

    score_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.xml', '.mxl', '.musicxml', '.mid', '.midi'))]
    if not score_files:
        print(f"[Tenuto Preprocess] No score files found in '{raw_dir}'.")
        return

    print(f"[Tenuto Preprocess] Found {len(score_files)} score files. Extracting 40D note features...")
    processed_count = 0
    skipped_count = 0

    for fname in tqdm(score_files, desc="Preprocessing"):
        raw_path = os.path.join(raw_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".pt"
        out_path = os.path.join(processed_dir, out_name)

        # Idempotency check: Skip if processed file exists and force flag is False
        if os.path.exists(out_path) and not force:
            skipped_count += 1
            continue

        features = extract_partitura_features(raw_path)
        if features is not None:
            torch.save({"x": features, "filename": fname}, out_path)
            processed_count += 1

    print(f"[Tenuto Preprocess] Complete! Processed: {processed_count}, Skipped (Already Processed): {skipped_count}.")

def main():
    parser = argparse.ArgumentParser(description="Tenuto Idempotent Dataset Preprocessor")
    parser.add_argument("--raw_dir", type=str, default="./data/raw", help="Directory containing raw MusicXML/MIDI files")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Output directory for .pt tensors")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if .pt tensors exist")
    args = parser.parse_args()

    preprocess_dataset(args.raw_dir, args.processed_dir, force=args.force)

if __name__ == "__main__":
    main()
