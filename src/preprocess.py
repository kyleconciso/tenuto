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

def preprocess_combined_dataset(data_dir: str = "./data", processed_dir: str = "./data/processed", force: bool = False, max_samples: int = None):
    train_out = os.path.join(processed_dir, "train")
    val_out = os.path.join(processed_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    global_idx = 0
    
    random.seed(42)

    pbar = tqdm(total=131342, desc="Preprocessing Score-Performance Pairs")

    # 1. Process ASAP
    asap_dir = os.path.join(data_dir, "asap")
    metadata_csv = os.path.join(asap_dir, "metadata.csv")
    if os.path.exists(metadata_csv):
        try:
            import pandas as pd
            df_asap = pd.read_csv(metadata_csv)
            for idx, row in df_asap.iterrows():
                if max_samples and processed_count + skipped_count >= max_samples:
                    break
                xml_rel = row.get('xml_score')
                midi_perf_rel = row.get('midi_performance')
                if pd.notna(xml_rel) and pd.notna(midi_perf_rel):
                    xml_path = os.path.join(asap_dir, xml_rel)
                    perf_path = os.path.join(asap_dir, midi_perf_rel)
                    if os.path.exists(xml_path) and os.path.exists(perf_path):
                        # Deterministic train/val split based on sample index
                        is_train = (global_idx % 100) < 85
                        dest_dir = train_out if is_train else val_out
                        dest_name = f"asap_pair_{global_idx:06d}.pt"
                        if os.path.exists(os.path.join(train_out, dest_name)) or os.path.exists(os.path.join(val_out, dest_name)):
                            if not force:
                                skipped_count += 1
                                global_idx += 1
                                pbar.update(1)
                                continue
                        if process_single_pair(xml_path, perf_path, "asap", dest_dir, dest_name, force):
                            processed_count += 1
                        else:
                            skipped_count += 1
                        global_idx += 1
                        pbar.update(1)
        except Exception as e:
            print(f"\n[TenutoPreprocess] ASAP metadata parsing note: {e}")
    elif os.path.exists(asap_dir):
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
                for pf in perf_files:
                    if max_samples and processed_count + skipped_count >= max_samples:
                        break
                    
                    is_train = random.random() < 0.85
                    dest_dir = train_out if is_train else val_out
                    dest_name = f"asap_pair_{global_idx:06d}.pt"
                    
                    if process_single_pair(score_file, pf, "asap", dest_dir, dest_name, force):
                        processed_count += 1
                    else:
                        skipped_count += 1
                    global_idx += 1
                    pbar.update(1)

    # 2. Process PianoCoRe directly from DataFrames
    pianocore_dir = os.path.join(data_dir, "pianocore")
    if os.path.exists(pianocore_dir):
        try:
            import pandas as pd
        except ImportError:
            pd = None
            print("\n[TenutoPreprocess] pandas not installed. Cannot parse PianoCoRe parquets.")
            
        for root, _, files in os.walk(pianocore_dir):
            for f in files:
                if max_samples and processed_count + skipped_count >= max_samples:
                    break
                if f.endswith('.parquet') and pd is not None:
                    pq_file = os.path.join(root, f)
                    try:
                        pbar.set_postfix_str(f"Loading {os.path.basename(pq_file)}")
                        df = pd.read_parquet(pq_file)
                        if 'tier_a_star' in df.columns:
                            df = df[df['tier_a_star'] == True]
                        
                        for idx, row in df.iterrows():
                            if max_samples and processed_count + skipped_count >= max_samples:
                                break
                                
                            xml_bytes = row.get('score_xml_bytes')
                            midi_bytes = row.get('performance_midi_bytes')
                            
                            if xml_bytes is not None and midi_bytes is not None:
                                is_train = (global_idx % 100) < 85
                                dest_dir = train_out if is_train else val_out
                                source_tag = f"pianocore_{row.get('id', idx)}"
                                dest_name = f"pianocore_pair_{global_idx:06d}.pt"
                                
                                if os.path.exists(os.path.join(train_out, dest_name)) or os.path.exists(os.path.join(val_out, dest_name)):
                                    if not force:
                                        skipped_count += 1
                                        global_idx += 1
                                        pbar.update(1)
                                        continue
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as xml_file:
                                    xml_file.write(xml_bytes)
                                    xml_path = xml_file.name
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
                                    mid_file.write(midi_bytes)
                                    mid_path = mid_file.name
                                
                                pbar.set_postfix_str(f"Processing chunk pair...")
                                if process_single_pair(xml_path, mid_path, source_tag, dest_dir, dest_name, force):
                                    processed_count += 1
                                else:
                                    skipped_count += 1
                                    
                                try:
                                    os.remove(xml_path)
                                    os.remove(mid_path)
                                except:
                                    pass
                                
                                global_idx += 1
                                pbar.update(1)
                                
                    except Exception as e:
                        print(f"\n[TenutoPreprocess] Failed to process {pq_file}: {e}")
                    
                    del df

    pbar.close()
    print(f"\n[TenutoPreprocess] Preprocessing Complete! Unified dataset saved to '{processed_dir}' ({processed_count} processed, {skipped_count} skipped).")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Combined ASAP + PianoCoRe Dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Processed output directory")
    parser.add_argument("--max_samples", type=int, default=10000, help="Optional max sample limit (default: 10000)")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    preprocess_combined_dataset(data_dir=args.data_dir, processed_dir=args.processed_dir, force=args.force, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
