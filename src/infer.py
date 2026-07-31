import os
import argparse
import torch
import numpy as np

from src.utils import get_device, load_checkpoint
from src.model import build_model
from src.features import extract_40d_features_from_score
from src.render import render_expressive_midi

def find_best_checkpoint(checkpoint_path: str = None):
    """Auto-detects the best available model checkpoint (JAX .msgpack or PyTorch .pth)."""
    gdrive_dir = "/content/drive/MyDrive/Tenuto/checkpoints"
    candidates = []
    if checkpoint_path:
        candidates.append(checkpoint_path)
    
    candidates.extend([
        os.path.join(gdrive_dir, "best_transformer_jax.msgpack"),
        os.path.join(gdrive_dir, "best_transformer_model.pth"),
        os.path.join(gdrive_dir, "latest_transformer_jax.msgpack"),
        os.path.join(gdrive_dir, "latest_transformer_model.pth"),
        "checkpoints/best_transformer_jax.msgpack",
        "checkpoints/best_transformer_model.pth",
        "checkpoints/latest_transformer_jax.msgpack",
        "checkpoints/latest_transformer_model.pth"
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

def infer_performance(score_path: str, checkpoint_path: str = None, model_type: str = "transformer", in_features: int = 40, output_midi: str = "output_expressive.mid"):
    """
    Universal Inference Script:
    Predicts expressive parameters using either JAX (.msgpack) or PyTorch (.pth) checkpoints.
    """
    device = get_device()
    
    # Locate score file
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

    # 1. Extract 40D Features
    print(f"[Tenuto Inference] Extracting note features from '{target_score}'...")
    features = extract_40d_features_from_score(target_score)
    
    # 2. Locate Checkpoint
    active_checkpoint = find_best_checkpoint(checkpoint_path)
    print(f"[Tenuto Inference] Active Checkpoint: '{active_checkpoint}'")

    predictions = None

    # Try JAX Inference if .msgpack checkpoint is found
    if active_checkpoint and active_checkpoint.endswith(".msgpack"):
        try:
            import jax
            import jax.numpy as jnp
            import flax
            from src.model_jax import TenutoTransformerJAX

            print(f"[Tenuto Inference] Loading JAX/Flax checkpoint from '{active_checkpoint}'...")
            jax_model = TenutoTransformerJAX(in_features=in_features)
            dummy_input = jnp.zeros((1, features.size(0), in_features), dtype=jnp.float32)
            init_params = jax_model.init(jax.random.PRNGKey(0), dummy_input, deterministic=True)['params']

            with open(active_checkpoint, "rb") as f:
                ckpt_bytes = f.read()
            restored = flax.serialization.msgpack_restore(ckpt_bytes)
            if isinstance(restored, dict) and "params" in restored:
                params = flax.serialization.from_state_dict(init_params, restored["params"])
            else:
                params = flax.serialization.from_state_dict(init_params, restored)

            x_jax = jnp.expand_dims(jnp.array(features.numpy()), axis=0)
            jax_preds = jax_model.apply({'params': params}, x_jax, deterministic=True)

            predictions = {
                "delta_t": torch.tensor(np.array(jax_preds["delta_t"][0])),
                "velocity": torch.tensor(np.array(jax_preds["velocity"][0])),
                "articulation": torch.tensor(np.array(jax_preds["articulation"][0])),
                "pedal": torch.tensor(np.array(jax_preds["pedal"][0])),
                "tempo_scale": torch.tensor(np.array(jax_preds["tempo_scale"][0]))
            }
            print("[Tenuto Inference] Successfully ran inference with JAX/Flax model!")
        except Exception as e:
            print(f"[Tenuto Inference] JAX loading failed ({e}). Falling back to PyTorch model.")

    # PyTorch Inference Fallback
    if predictions is None:
        print(f"[Tenuto Inference] Building PyTorch model & loading weights...")
        x = features.unsqueeze(0).to(device)
        model = build_model(model_name=model_type, in_features=in_features).to(device)
        if active_checkpoint and os.path.exists(active_checkpoint) and active_checkpoint.endswith((".pth", ".pt")):
            load_checkpoint(active_checkpoint, model)
        model.eval()
        with torch.no_grad():
            preds_raw = model(x)
            predictions = {k: v.squeeze(0).cpu() for k, v in preds_raw.items()}

    print("[Tenuto Inference] Predictions summary:")
    print(f"  • Micro Timing Shift (\\Delta t range): [{predictions['delta_t'].min().item():.4f}s, {predictions['delta_t'].max().item():.4f}s]")
    print(f"  • Velocity range:                      [{predictions['velocity'].min().item():.1f}, {predictions['velocity'].max().item():.1f}]")
    if "tempo_scale" in predictions:
        print(f"  • Tempo Scale range:                  [{predictions['tempo_scale'].min().item():.2f}x, {predictions['tempo_scale'].max().item():.2f}x]")

    # 3. Render Expressive MIDI File
    try:
        import partitura as pt
        score = pt.load_score(target_score)
        score_notes_raw = score.note_array()
        
        score_notes = []
        for i in range(len(score_notes_raw)):
            note = score_notes_raw[i]
            onset = note['onset_sec'] if 'onset_sec' in score_notes_raw.dtype.names else (note['onset_beat'] * 0.5 if 'onset_beat' in score_notes_raw.dtype.names else i * 0.25)
            dur = note['duration_sec'] if 'duration_sec' in score_notes_raw.dtype.names else (note['duration_beat'] * 0.5 if 'duration_beat' in score_notes_raw.dtype.names else 0.25)
            pitch = note['pitch'] if 'pitch' in score_notes_raw.dtype.names else 60
            score_notes.append({
                "pitch": pitch,
                "onset_sec": onset,
                "duration_sec": dur
            })
    except Exception as e:
        print(f"[Tenuto Inference] Score note loading info: {e}. Generating target note sequence.")
        score_notes = [{"pitch": 60 + (i % 24), "onset_sec": i * 0.25, "duration_sec": 0.25} for i in range(features.size(0))]

    render_expressive_midi(score_notes, predictions, output_midi_path=output_midi)
    return predictions

def main(args=None):
    parser = argparse.ArgumentParser(description="Tenuto Expressive Performance Inference Engine")
    parser.add_argument("--score", type=str, default="data/asap/Balakirev/Islamey/xml_score.musicxml", help="Input MusicXML score file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights (JAX .msgpack or PyTorch .pth)")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bigru"])
    parser.add_argument("--in_features", type=int, default=40, help="Feature dimension (40 or 10)")
    parser.add_argument("--output_midi", type=str, default="output_expressive.mid", help="Output MIDI file path")
    parsed_args = parser.parse_args(args)
    
    infer_performance(
        parsed_args.score,
        parsed_args.checkpoint,
        model_type=parsed_args.model_type,
        in_features=parsed_args.in_features,
        output_midi=parsed_args.output_midi
    )

if __name__ == "__main__":
    main()
