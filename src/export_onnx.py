import os
import argparse
import torch

from src.model import build_model

def export_to_onnx(model_path: str, output_onnx_path: str = "tenuto_transformer.onnx", in_features: int = 40, seq_len: int = 256):
    """
    Exports trained Tenuto Transformer model to ONNX format for sub-50ms execution.
    """
    print(f"[TenutoONNX] Loading PyTorch model for ONNX export...")
    model = build_model(model_name="transformer", in_features=in_features)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        print(f"[TenutoONNX] Loaded weights from '{model_path}'.")
    else:
        print(f"[TenutoONNX] Model checkpoint '{model_path}' not found. Exporting model shell.")

    model.eval()
    dummy_input = torch.randn(1, seq_len, in_features, dtype=torch.float32)

    print(f"[TenutoONNX] Exporting to '{output_onnx_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["note_features"],
        output_names=["tempo_scale", "dynamic_base", "delta_t", "velocity", "articulation", "pedal"],
        dynamic_axes={
            "note_features": {0: "batch_size", 1: "seq_len"},
            "delta_t": {0: "batch_size", 1: "seq_len"},
            "velocity": {0: "batch_size", 1: "seq_len"}
        }
    )
    print(f"[TenutoONNX] Successfully exported ONNX model to '{output_onnx_path}'!")

def main():
    parser = argparse.ArgumentParser(description="Export Tenuto Model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_transformer_model.pth", help="PyTorch checkpoint file")
    parser.add_argument("--output", type=str, default="tenuto_transformer.onnx", help="Output ONNX filename")
    parser.add_argument("--in_features", type=int, default=40, help="Input note feature dimension")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, in_features=args.in_features)

if __name__ == "__main__":
    main()
