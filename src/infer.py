import os
import argparse
import torch
import numpy as np

from src.utils import get_device, load_checkpoint
from src.model import build_model
from src.features import extract_40d_features_from_score
from src.render import render_expressive_midi

def infer_performance(score_path: str, checkpoint_path: str, model_type: str = "transformer", in_features: int = 40, output_midi: str = "output_expressive.mid"):
    """
    Inference script: Predicts expressive parameters and renders an expressive MIDI file.
    """
    device = get_device()
    
    # Check if score_path exists; if not, search for any available score file in data/
    target_score = score_path
    if not os.path.exists(target_score):
        found = False
        for root, _, files in os.walk("./data"):
            for f in files:
                if f.lower().endswith(('.xml', '.mxl', '.musicxml', '.mid', '.midi')):
                    target_score = os.path.join(root, f)
                    print(f"[Tenuto Inference] Score '{score_path}' not found. Using dataset score '{target_score}'.")
                    found = True
                    break
            if found:
                break

    # 1. Feature Extraction
    print(f"[Tenuto Inference] Extracting note features from '{target_score}'...")
    features = extract_40d_features_from_score(target_score)
    x = features.unsqueeze(0).to(device) # Batch dimension (1, SeqLen, InFeatures)

    # 2. Load Model & Weights
    print(f"[Tenuto Inference] Building {model_type} model & loading weights from '{checkpoint_path}'...")
    model = build_model(model_name=model_type, in_features=in_features).to(device)
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
    else:
        print(f"[Tenuto Inference] Checkpoint '{checkpoint_path}' not found. Using initialized weights.")

    model.eval()
    with torch.no_grad():
        predictions = model(x)

    print("[Tenuto Inference] Predictions complete:")
    print(f"  - Micro Timing Shift (\\Delta t range): [{predictions['delta_t'].min().item():.4f}s, {predictions['delta_t'].max().item():.4f}s]")
    print(f"  - Velocity range: [{predictions['velocity'].min().item():.1f}, {predictions['velocity'].max().item():.1f}]")
    if "tempo_scale" in predictions:
        print(f"  - Tempo Scale range: [{predictions['tempo_scale'].min().item():.2f}x, {predictions['tempo_scale'].max().item():.2f}x]")

    # 3. Render Expressive MIDI File
    score_notes_placeholder = [{"pitch": 60 + (i % 24), "onset_sec": i * 0.25, "duration_sec": 0.25} for i in range(x.size(1))]
    render_expressive_midi(score_notes_placeholder, predictions, output_midi_path=output_midi)

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Tenuto Expressive Performance Inference Engine")
    parser.add_argument("--score", type=str, default="sample.xml", help="Input MusicXML score file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_transformer_model.pth", help="Path to model weights")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bigru"])
    parser.add_argument("--in_features", type=int, default=40, help="Feature dimension (40 or 10)")
    parser.add_argument("--output_midi", type=str, default="output_expressive.mid", help="Output MIDI file path")
    args = parser.parse_args()

    infer_performance(args.score, args.checkpoint, model_type=args.model_type, in_features=args.in_features, output_midi=args.output_midi)

if __name__ == "__main__":
    main()
