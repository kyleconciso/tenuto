import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from tqdm import tqdm
import tempfile

from src.features import extract_40d_features_from_score
from src.alignment import compute_alignment_targets

def process_single_pair(score_path, perf_path, source_tag, dest_dir, dest_name, force):
    dest_path = os.path.join(dest_dir, dest_name)
    if os.path.exists(dest_path) and not force:
        return False # Skipped

    x = extract_40d_features_from_score(score_path)
    targets = compute_alignment_targets(score_path, perf_path)

    if x is not None and len(x) > 0 and targets is not None:
        torch.save({
            "x": x,
            "targets": targets,
            "score_path": score_path,
            "perf_path": perf_path,
            "source": source_tag
        }, dest_path)
        return True # Processed
    return False

def worker_task(args):
    score_path, perf_path, source_tag, dest_dir, dest_name, force = args
    try:
        success = process_single_pair(score_path, perf_path, source_tag, dest_dir, dest_name, force)
        return success
    except Exception:
        return False

def preprocess_combined_dataset(data_dir: str = "./data", processed_dir: str = "./storage/processed", force: bool = False, max_samples: int = 10000, num_workers: int = None):
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)
    
    train_out = os.path.join(processed_dir, "train")
    val_out = os.path.join(processed_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    global_idx = 0

    print(f"[TenutoPreprocess] Collecting sample pairs across datasets...")

    # 1. Collect ASAP pairs
    asap_dir = os.path.join(data_dir, "asap")
    metadata_csv = os.path.join(asap_dir, "metadata.csv")
    if os.path.exists(metadata_csv):
        try:
            import pandas as pd
            df_asap = pd.read_csv(metadata_csv)
            for idx, row in df_asap.iterrows():
                if max_samples and len(tasks) >= max_samples:
                    break
                xml_rel = row.get('xml_score')
                midi_perf_rel = row.get('midi_performance')
                if pd.notna(xml_rel) and pd.notna(midi_perf_rel):
                    xml_path = os.path.join(asap_dir, xml_rel)
                    perf_path = os.path.join(asap_dir, midi_perf_rel)
                    if os.path.exists(xml_path) and os.path.exists(perf_path):
                        is_train = (global_idx % 100) < 85
                        dest_dir = train_out if is_train else val_out
                        dest_name = f"asap_pair_{global_idx:06d}.pt"
                        tasks.append((xml_path, perf_path, "asap", dest_dir, dest_name, force))
                        global_idx += 1
        except Exception as e:
            print(f"[TenutoPreprocess] ASAP metadata parsing note: {e}")

    # 2. Collect PianoCoRe pairs from Parquet
    pianocore_dir = os.path.join(data_dir, "pianocore")
    if os.path.exists(pianocore_dir):
        try:
            import pandas as pd
        except ImportError:
            pd = None

        for root, _, files in os.walk(pianocore_dir):
            for f in files:
                if max_samples and len(tasks) >= max_samples:
                    break
                if f.endswith('.parquet') and pd is not None:
                    pq_file = os.path.join(root, f)
                    try:
                        df = pd.read_parquet(pq_file)
                        if 'tier_a_star' in df.columns:
                            df = df[df['tier_a_star'] == True]
                        
                        for idx, row in df.iterrows():
                            if max_samples and len(tasks) >= max_samples:
                                break
                            xml_bytes = row.get('score_xml_bytes')
                            midi_bytes = row.get('performance_midi_bytes')
                            if xml_bytes is not None and midi_bytes is not None:
                                is_train = (global_idx % 100) < 85
                                dest_dir = train_out if is_train else val_out
                                source_tag = f"pianocore_{row.get('id', idx)}"
                                dest_name = f"pianocore_pair_{global_idx:06d}.pt"
                                
                                # Write persistent temp files for multiprocessing workers
                                tmp_xml = os.path.join(tempfile.gettempdir(), f"pc_xml_{global_idx:06d}.xml")
                                tmp_mid = os.path.join(tempfile.gettempdir(), f"pc_mid_{global_idx:06d}.mid")
                                with open(tmp_xml, "wb") as xf:
                                    xf.write(xml_bytes)
                                with open(tmp_mid, "wb") as mf:
                                    mf.write(midi_bytes)
                                
                                tasks.append((tmp_xml, tmp_mid, source_tag, dest_dir, dest_name, force))
                                global_idx += 1
                    except Exception as e:
                        print(f"[TenutoPreprocess] PianoCoRe error: {e}")

    print(f"[TenutoPreprocess] Ready to process {len(tasks)} sample pairs using {num_workers} parallel CPU workers.")

    processed_count = 0
    skipped_count = 0

    pbar = tqdm(total=len(tasks), desc=f"Multiprocess Preprocessing ({num_workers} Workers)")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                processed_count += 1
            else:
                skipped_count += 1
            pbar.update(1)

    pbar.close()
    print(f"\n[TenutoPreprocess] Preprocessing Complete! Unified dataset saved to '{processed_dir}' ({processed_count} processed, {skipped_count} skipped).")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Combined ASAP + PianoCoRe Dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--processed_dir", type=str, default="./storage/processed", help="Processed output directory (default: ./storage/processed)")
    parser.add_argument("--max_samples", type=int, default=10000, help="Optional max sample limit (default: 10000)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel CPU worker processes")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    preprocess_combined_dataset(data_dir=args.data_dir, processed_dir=args.processed_dir, force=args.force, max_samples=args.max_samples, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
