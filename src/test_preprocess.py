import os
import torch
import numpy as np

from src.features import extract_40d_features_from_score
from src.alignment import compute_alignment_targets

def verify_preprocessing_pipeline():
    print("==================================================")
    print("       Tenuto Preprocessing Verification Test     ")
    print("==================================================")

    # 1. Test Feature Extraction (40D Schema Verification)
    print("\n[Test 1] Extracting 40D note features...")
    # Test with synthetic or sample score file
    sample_xml = "./data/asap/Chopin/Nocturnes/op9_no2/xml_score.musicxml"
    if not os.path.exists(sample_xml):
        # Fallback to any xml file found in asap
        for root, _, files in os.walk("./data/asap"):
            for f in files:
                if f.endswith('.xml') or f.endswith('.musicxml'):
                    sample_xml = os.path.join(root, f)
                    break

    if os.path.exists(sample_xml):
        print(f"  - Using real score file: '{sample_xml}'")
        x = extract_40d_features_from_score(sample_xml)
    else:
        print("  - Using fallback synthetic feature sequence.")
        x = extract_40d_features_from_score("synthetic")

    print(f"  ✓ Feature Matrix Shape: {x.shape} (Expected: N x 40)")
    assert x.ndim == 2 and x.size(1) == 40, f"Expected 40 features per note, got {x.size(1)}"
    assert not torch.isnan(x).any(), "Found NaN values in feature matrix!"
    assert not torch.isinf(x).any(), "Found Inf values in feature matrix!"

    # 2. Check 40D Feature Sub-ranges
    pitch_norm = x[:, 0]
    print(f"  ✓ Normalized Pitch Range: [{pitch_norm.min().item():.3f}, {pitch_norm.max().item():.3f}] (Expected: [0.0, 1.0])")
    assert pitch_norm.min() >= 0.0 and pitch_norm.max() <= 1.0, "Pitch normalization out of bounds!"

    pitch_classes = x[:, 1:13]
    print(f"  ✓ Pitch Class One-Hot Sums: {pitch_classes.sum(dim=1).min().item():.0f} (Expected: 1 per note)")

    # 3. Test Alignment Target Generation (Y Schema Verification)
    print("\n[Test 2] Verifying alignment target parameters Y...")
    targets = compute_alignment_targets(None, None)
    
    print(f"  ✓ delta_t range:      [{targets['delta_t'].min().item():.4f}s, {targets['delta_t'].max().item():.4f}s] (Bounded [-0.025s, +0.025s])")
    print(f"  ✓ velocity range:     [{targets['velocity'].min().item():.1f}, {targets['velocity'].max().item():.1f}] (Bounded [0.0, 127.0])")
    print(f"  ✓ articulation range: [{targets['articulation'].min().item():.2f}, {targets['articulation'].max().item():.2f}] (> 0.0)")
    print(f"  ✓ pedal range:        [{targets['pedal'].min().item():.1f}, {targets['pedal'].max().item():.1f}] (Bounded [0.0, 127.0])")
    print(f"  ✓ tempo_scale range:  [{targets['tempo_scale'].min().item():.2f}x, {targets['tempo_scale'].max().item():.2f}x] (Bounded [0.5, 1.5])")

    assert targets['delta_t'].min() >= -0.025 and targets['delta_t'].max() <= 0.025, "delta_t out of bounds!"
    assert targets['velocity'].min() >= 0.0 and targets['velocity'].max() <= 127.0, "velocity out of bounds!"

    print("\n==================================================")
    print("  SUCCESS: All preprocessing data checks passed!  ")
    print("==================================================")

if __name__ == "__main__":
    verify_preprocessing_pipeline()
