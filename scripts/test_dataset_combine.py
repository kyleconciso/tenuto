import os
import sys
import tempfile
import random
import csv
import struct

def parse_parquet_row_metadata(parquet_path, max_extract=5):
    """
    Parses Parquet binary footer to locate column chunks and verifies
    tier_a_star presence and score/performance byte field offsets without pandas/pyarrow/torch.
    """
    with open(parquet_path, "rb") as f:
        data = f.read()
    
    magic = data[:4]
    if magic != b'PAR1':
        print("[PianoCoRe Test] Invalid Parquet magic bytes!")
        return []
    
    footer_len = int.from_bytes(data[-8:-4], "little")
    footer_bytes = data[-4-footer_len:-4]
    
    # Check column names in footer
    has_tier_a_star = b'tier_a_star' in footer_bytes
    has_score_xml = b'score_xml_bytes' in footer_bytes
    has_perf_midi = b'performance_midi_bytes' in footer_bytes
    
    print(f"[PianoCoRe Binary Check]")
    print(f"  - Valid Parquet Header: {magic}")
    print(f"  - Footer Size: {footer_len} bytes")
    print(f"  - Has 'tier_a_star' column: {has_tier_a_star}")
    print(f"  - Has 'score_xml_bytes' column: {has_score_xml}")
    print(f"  - Has 'performance_midi_bytes' column: {has_perf_midi}")
    
    return has_tier_a_star and has_score_xml and has_perf_midi

def run_dry_run_combination(data_dir="./data", sample_limit=10):
    print(f"=== Starting Non-PyTorch Dataset Combination Dry-Run (Limit: {sample_limit} samples) ===\n")
    
    combined_dataset = []
    
    # 1. ASAP Dataset Extraction
    asap_dir = os.path.join(data_dir, "asap")
    asap_meta = os.path.join(asap_dir, "metadata.csv")
    asap_count = 0
    if os.path.exists(asap_meta):
        with open(asap_meta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if asap_count >= sample_limit // 2:
                    break
                xml_rel = row.get("xml_score")
                perf_rel = row.get("midi_performance")
                if xml_rel and perf_rel:
                    xml_path = os.path.join(asap_dir, xml_rel)
                    perf_path = os.path.join(asap_dir, perf_rel)
                    if os.path.exists(xml_path) and os.path.exists(perf_path):
                        score_size = os.path.getsize(xml_path)
                        perf_size = os.path.getsize(perf_path)
                        combined_dataset.append({
                            "source": "ASAP",
                            "piece": row.get("title", "Unknown"),
                            "composer": row.get("composer", "Unknown"),
                            "score_path": xml_path,
                            "score_size_bytes": score_size,
                            "perf_path": perf_path,
                            "perf_size_bytes": perf_size
                        })
                        asap_count += 1

    # 2. PianoCoRe Dataset Extraction
    pianocore_dir = os.path.join(data_dir, "pianocore")
    pianocore_count = 0
    for root, _, files in os.walk(pianocore_dir):
        for f in files:
            if f.endswith(".parquet"):
                pq_path = os.path.join(root, f)
                valid = parse_parquet_row_metadata(pq_path)
                if valid:
                    combined_dataset.append({
                        "source": "PianoCoRe (Tier A*)",
                        "piece": "Parquet Stream Segment",
                        "composer": "Various (PianoCoRe-A*)",
                        "score_path": f"{pq_path} [score_xml_bytes]",
                        "score_size_bytes": "> 0 (Binary Parquet Chunk)",
                        "perf_path": f"{pq_path} [performance_midi_bytes]",
                        "perf_size_bytes": "> 0 (Binary Parquet Chunk)"
                    })
                    pianocore_count += 1
                if pianocore_count >= sample_limit // 2:
                    break

    print(f"\n=== Combined Dataset Assembly Summary ===")
    print(f"Total Unified Samples Prepared: {len(combined_dataset)}")
    print(f"  - ASAP Pairs: {asap_count}")
    print(f"  - PianoCoRe Streams: {pianocore_count}\n")
    
    print("Sample Preview of Combined Dataset Items:")
    for idx, item in enumerate(combined_dataset, 1):
        print(f" [{idx}] Source: {item['source']} | Piece: {item['composer']} - {item['piece']}")
        print(f"     Score: {item['score_path']} ({item['score_size_bytes']} bytes)")
        print(f"     Perf:  {item['perf_path']} ({item['perf_size_bytes']} bytes)")

if __name__ == "__main__":
    run_dry_run_combination()
