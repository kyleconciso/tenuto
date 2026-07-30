import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm

from src.features import extract_40d_features_from_score
from src.alignment import compute_alignment_targets

def preprocess_combined_dataset(data_dir: str = "./data", processed_dir: str = "./data/processed", force: bool = False, max_samples: int = None):
    """
    Combines score-performance pairs from both ASAP dataset and PianoCoRe-A.
    
    Each score XML/MIDI file in ASAP has multiple aligned human performances (e.g. 5-15 performances per piece).
    This function extracts note features X and matches them with every aligned performance pair Y.
    """
    train_out = os.path.join(processed_dir, "train")
    val_out = os.path.join(processed_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    aligned_pairs = []

    # 1. Scan ASAP dataset for score + performance alignment pairs
    asap_dir = os.path.join(data_dir, "asap")
    if os.path.exists(asap_dir):
        for root, _, files in os.walk(asap_dir):
            score_file = None
            perf_files = []
            for f in files:
                f_lower = f.lower()
                if f_lower.endswith(('.xml', '.mxl', '.musicxml')) or (f_lower.endswith('.mid') and 'score' in f_lower):
                    score_file = os.path.join(root, f)
                elif f_lower.endswith('.mid') and not 'score' in f_lower:
                    perf_files.append(os.path.join(root, f))
            
            if score_file:
                if perf_files:
                    for pf in perf_files:
                        aligned_pairs.append((score_file, pf, "asap"))
                else:
                    aligned_pairs.append((score_file, None, "asap"))

    # 2. Scan PianoCoRe dataset for aligned score-performance pairs
    pianocore_dir = os.path.join(data_dir, "pianocore")
    if os.path.exists(pianocore_dir):
        for root, _, files in os.walk(pianocore_dir):
            score_file = None
            perf_files = []
            for f in files:
                f_lower = f.lower()
                if 'score' in f_lower and f_lower.endswith(('.mid', '.midi', '.xml')):
                    score_file = os.path.join(root, f)
                elif f_lower.endswith(('.mid', '.midi')) and not 'score' in f_lower:
                    perf_files.append(os.path.join(root, f))

            if score_file and perf_files:
                for pf in perf_files:
                    aligned_pairs.append((score_file, pf, "pianocore"))
            elif score_file:
                aligned_pairs.append((score_file, None, "pianocore"))

    if not aligned_pairs:
        print(f"[TenutoPreprocess] No score-performance pairs found in '{data_dir}'.")
        return

    if max_samples and len(aligned_pairs) > max_samples:
        print(f"[TenutoPreprocess] Cap requested: limiting dataset to {max_samples} pairs out of {len(aligned_pairs)} found.")
        aligned_pairs = aligned_pairs[:max_samples]

    print(f"[TenutoPreprocess] Found {len(aligned_pairs)} total aligned score-performance pairs. Processing...")

    # 85/15 Train/Val Split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(aligned_pairs), generator=generator).tolist()
    split_idx = int(0.85 * len(aligned_pairs))

    processed_count = 0
    skipped_count = 0

    for i, idx in enumerate(tqdm(indices, desc="Preprocessing Score-Performance Pairs")):
        score_path, perf_path, source_tag = aligned_pairs[idx]
        is_train = i < split_idx
        dest_dir = train_out if is_train else val_out

        dest_name = f"{source_tag}_pair_{i:05d}.pt"
        dest_path = os.path.join(dest_dir, dest_name)

        if os.path.exists(dest_path) and not force:
            skipped_count += 1
            continue

        x = extract_40d_features_from_score(score_path)
        targets = compute_alignment_targets(score_path, perf_path)

        if x is not None and len(x) > 0:
            torch.save({
                "x": x,
                "targets": targets,
                "score_path": score_path,
                "perf_path": perf_path,
                "source": source_tag
            }, dest_path)
            processed_count += 1

    print(f"[TenutoPreprocess] Preprocessing Complete! Unified dataset saved to '{processed_dir}' ({processed_count} processed, {skipped_count} skipped).")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Combined ASAP + PianoCoRe Dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Processed output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max sample limit")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    preprocess_combined_dataset(data_dir=args.data_dir, processed_dir=args.processed_dir, force=args.force, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
