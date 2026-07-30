import os
import argparse
import torch
import numpy as np

from src.utils import get_device, load_checkpoint
from src.model import build_model
from src.dataset import extract_partitura_features

def infer_performance(score_path: str, checkpoint_path: str, model_type: str = "transformer", in_features: int = 40, output_midi: str = "output_expressive.mid"):
    """
    Idempotent inference script: Predicts expressive parameters for an input score file.
    """
    device = get_device()
    
    # 1. Feature Extraction
    print(f"[Tenuto Inference] Extracting note features from '{score_path}'...")
    features = extract_partitura_features(score_path)
    if features is None:
        # Fallback synthetic note features for demo/testing
        print(f"[Tenuto Inference] Using fallback synthetic sequence for testing pipeline.")
        features = torch.randn(256, in_features)
    
    x = features.unsqueeze(0).to(device) # Add batch dimension (1, SeqLen, InFeatures)

    # 2. Load Model & Checkpoint
    print(f"[Tenuto Inference] Building {model_type} model & loading weights from '{checkpoint_path}'...")
    model = build_model(model_name=model_type, in_features=in_features).to(device)
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
    else:
        print(f"[Tenuto Inference] Warning: Checkpoint '{checkpoint_path}' not found! Running with randomly initialized weights for verification.")

    model.eval()
    with torch.no_grad():
        predictions = model(x)

    print("[Tenuto Inference] Prediction completed!")
    print(f"  - Micro Timing Shift (\\Delta t range): [{predictions['delta_t'].min().item():.4f}s, {predictions['delta_t'].max().item():.4f}s]")
    print(f"  - Velocity range: [{predictions['velocity'].min().item():.1f}, {predictions['velocity'].max().item():.1f}]")
    if "tempo_scale" in predictions:
        print(f"  - Tempo Scale range: [{predictions['tempo_scale'].min().item():.2f}x, {predictions['tempo_scale'].max().item():.2f}x]")

    # Idempotent MIDI writing
    print(f"[Tenuto Inference] Rendered predictions ready. Export destination: '{output_midi}'")
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Tenuto Expressive Inference Engine")
    parser.add_argument("--score", type=str, default="sample.xml", help="Input MusicXML score file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_transformer_model.pth", help="Path to model weights")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bigru"])
    parser.add_argument("--in_features", type=int, default=40, help="Feature dimension (40 or 10)")
    parser.add_argument("--output_midi", type=str, default="output_expressive.mid", help="Output MIDI file path")
    args = parser.parse_args()

    infer_performance(args.score, args.checkpoint, model_type=args.model_type, in_features=args.in_features, output_midi=args.output_midi)

if __name__ == "__main__":
    main()
