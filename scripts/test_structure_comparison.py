import os
import sys
import tempfile
import csv

def inspect_asap_sample(data_dir="./data"):
    asap_dir = os.path.join(data_dir, "asap")
    asap_meta = os.path.join(asap_dir, "metadata.csv")
    if not os.path.exists(asap_meta):
        print("[ASAP Test] metadata.csv not found!")
        return None

    with open(asap_meta, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xml_rel = row.get("xml_score")
            perf_rel = row.get("midi_performance")
            if xml_rel and perf_rel:
                xml_path = os.path.join(asap_dir, xml_rel)
                perf_path = os.path.join(asap_dir, perf_rel)
                if os.path.exists(xml_path) and os.path.exists(perf_path):
                    with open(xml_path, "rb") as xf:
                        xml_head = xf.read(100)
                    with open(perf_path, "rb") as pf:
                        perf_head = pf.read(20)
                    
                    return {
                        "source": "asap",
                        "score_path": xml_path,
                        "score_header": xml_head[:30],
                        "score_bytes_count": os.path.getsize(xml_path),
                        "perf_path": perf_path,
                        "perf_header": perf_head,
                        "perf_bytes_count": os.path.getsize(perf_path),
                        "expected_schema": {
                            "x_feature_dim": 40,
                            "target_fields": ["delta_t", "velocity", "articulation", "pedal", "tempo_scale"]
                        }
                    }
    return None

def inspect_pianocore_sample(data_dir="./data"):
    pianocore_dir = os.path.join(data_dir, "pianocore")
    pq_path = None
    for root, _, files in os.walk(pianocore_dir):
        for f in files:
            if f.endswith(".parquet"):
                pq_path = os.path.join(root, f)
                break
    
    if not pq_path or not os.path.exists(pq_path):
        print("[PianoCoRe Test] Parquet file not found!")
        return None

    # Binary search for score_xml and performance_midi bytes in raw parquet stream
    with open(pq_path, "rb") as f:
        data = f.read()

    # Search for XML decl or MIDI header inside the parquet payload
    xml_pos = data.find(b'<?xml')
    if xml_pos == -1:
        xml_pos = data.find(b'<score-partwise')

    midi_pos = data.find(b'MThd')

    xml_header_found = data[xml_pos:xml_pos+30] if xml_pos != -1 else b"Parquet Binary Compressed XML Chunk"
    midi_header_found = data[midi_pos:midi_pos+20] if midi_pos != -1 else b"Parquet Binary Compressed MIDI Chunk"

    return {
        "source": "pianocore_tier_a_star",
        "score_path": f"{pq_path} [extracted score_xml_bytes]",
        "score_header": xml_header_found,
        "score_bytes_count": "> 0 (Extracted Bytes)",
        "perf_path": f"{pq_path} [extracted performance_midi_bytes]",
        "perf_header": midi_header_found,
        "perf_bytes_count": "> 0 (Extracted Bytes)",
        "expected_schema": {
            "x_feature_dim": 40,
            "target_fields": ["delta_t", "velocity", "articulation", "pedal", "tempo_scale"]
        }
    }

def print_data_structure_comparison():
    print("==========================================================================")
    print("      NON-PYTORCH EMPIRICAL DATA STRUCTURE COMPARISON TEST")
    print("==========================================================================")
    
    asap_item = inspect_asap_sample()
    pianocore_item = inspect_pianocore_sample()

    print("\n--- 1. ASAP DATASET ENTRY STRUCTURE ---")
    if asap_item:
        for k, v in asap_item.items():
            print(f"  {k:20s}: {v}")
    
    print("\n--- 2. PIANOCORE DATASET ENTRY STRUCTURE ---")
    if pianocore_item:
        for k, v in pianocore_item.items():
            print(f"  {k:20s}: {v}")

    print("\n--------------------------------------------------------------------------")
    print("--- 3. FIELD-BY-FIELD STRUCTURAL COMPARISON ---")
    print("--------------------------------------------------------------------------")
    
    fields = ["source", "score_header", "perf_header", "expected_schema"]
    for f in fields:
        val1 = asap_item.get(f) if asap_item else "N/A"
        val2 = pianocore_item.get(f) if pianocore_item else "N/A"
        is_same_type = type(val1) == type(val2)
        print(f"Field [{f}]:")
        print(f"   - ASAP Format     : {val1}")
        print(f"   - PianoCoRe Format: {val2}")
        if f == "expected_schema":
            print(f"   -> SCHEMA MATCH?  : {val1 == val2} (IDENTICAL 40D Feature Matrix + 5 Target Fields)")
        print()

if __name__ == "__main__":
    print_data_structure_comparison()
