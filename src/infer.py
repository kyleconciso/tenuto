import os
import argparse
import torch
import numpy as np

from src.utils import get_device, load_checkpoint
from src.model import build_model
from src.features import extract_40d_features_from_score
from src.render import render_expressive_midi

def find_best_checkpoint(checkpoint_path: str = None, preferred_ext: str = None):
    """Auto-detects the best available model checkpoint (JAX .msgpack or PyTorch .pth)."""
    gdrive_dir = "/content/drive/MyDrive/Tenuto/checkpoints"
    candidates = []
    if checkpoint_path:
        candidates.append(checkpoint_path)
    
    all_defaults = [
        os.path.join(gdrive_dir, "best_transformer_jax.msgpack"),
        os.path.join(gdrive_dir, "best_transformer_model.pth"),
        os.path.join(gdrive_dir, "latest_transformer_jax.msgpack"),
        os.path.join(gdrive_dir, "latest_transformer_model.pth"),
        "checkpoints/best_transformer_jax.msgpack",
        "checkpoints/best_transformer_model.pth",
        "checkpoints/latest_transformer_jax.msgpack",
        "checkpoints/latest_transformer_model.pth"
    ]
    
    if preferred_ext:
        candidates.extend([p for p in all_defaults if p.endswith(preferred_ext)])
    candidates.extend(all_defaults)

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
    
    # Locate score file or preprocessed data file
    target_score = score_path
    if not os.path.exists(target_score):
        found = False
        search_roots = ["./data", "data", "/content/tenuto/data"]
        for sroot in search_roots:
            if os.path.exists(sroot):
                for root, _, files in os.walk(sroot):
                    for f in files:
                        if f.lower().endswith(('.xml', '.mxl', '.musicxml', '.mid', '.midi', '.pt')):
                            target_score = os.path.join(root, f)
                            print(f"[Tenuto Inference] Score '{score_path}' not found. Using dataset file '{target_score}'.")
                            found = True
                            break
                    if found:
                        break
            if found:
                break

    # 1. Extract 40D Features
    features = None
    if os.path.exists(target_score):
        if target_score.endswith('.pt'):
            try:
                pt_data = torch.load(target_score, map_location="cpu")
                features = pt_data.get("x") if isinstance(pt_data, dict) else pt_data
                if features is not None and features.ndim == 1:
                    features = features.unsqueeze(0)
                print(f"[Tenuto Inference] Loaded preprocessed feature matrix {list(features.shape)} from '{target_score}'.")
            except Exception as e:
                print(f"[Tenuto Inference] Notice: could not load preprocessed tensor from '{target_score}': {e}")
        else:
            print(f"[Tenuto Inference] Extracting note features from '{target_score}'...")
            features = extract_40d_features_from_score(target_score)

    if features is None or features.numel() == 0:
        # Secondary fallback: look for any .pt preprocessed file
        for sroot in ["./data", "data", "/content/tenuto/data"]:
            if os.path.exists(sroot):
                for root, _, files in os.walk(sroot):
                    for f in files:
                        if f.endswith('.pt'):
                            pt_path = os.path.join(root, f)
                            try:
                                pt_data = torch.load(pt_path, map_location="cpu")
                                pt_x = pt_data.get("x") if isinstance(pt_data, dict) else pt_data
                                if pt_x is not None and pt_x.numel() > 0:
                                    features = pt_x if pt_x.ndim > 1 else pt_x.unsqueeze(0)
                                    target_score = pt_path
                                    print(f"[Tenuto Inference] Using preprocessed dataset tensor from '{pt_path}'.")
                                    break
                            except Exception:
                                pass
                    if features is not None:
                        break
            if features is not None:
                break

    if features is None or features.numel() == 0:
        print(f"[Tenuto Inference] Score '{target_score}' missing or invalid. Falling back to synthetic score note features (64 notes)...")
        num_notes = 64
        features = torch.zeros((num_notes, in_features), dtype=torch.float32)
        features[:, 0] = torch.tensor([(60 + (i % 24)) / 127.0 for i in range(num_notes)])
        features[:, 15] = 0.25 # duration
        features[:, 21] = 0.5  # tempo
        features[:, 26] = 1.0  # dynamic marking (mf)
        features[:, 38] = 1.0  # staff 1
    
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
        
        pth_ckpt = active_checkpoint if (active_checkpoint and active_checkpoint.endswith((".pth", ".pt"))) else find_best_checkpoint(preferred_ext=".pth")
        if pth_ckpt and os.path.exists(pth_ckpt) and pth_ckpt.endswith((".pth", ".pt")):
            load_checkpoint(pth_ckpt, model)
            
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
    score_notes = None
    if os.path.exists(target_score) and not target_score.endswith('.pt'):
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
            print(f"[Tenuto Inference] Partitura note loading note: {e}.")

    if score_notes is None:
        print("[Tenuto Inference] Reconstructing note sequence from feature matrix (max 256 notes)...")
        num_notes = min(features.size(0), 256)
        score_notes = []
        curr_onset = 0.0
        for i in range(num_notes):
            raw_pitch = float(features[i, 0].item()) * 127.0 if features.size(1) > 0 else 60.0
            pitch = int(np.clip(round(raw_pitch), 21, 108)) if raw_pitch > 10 else int(60 + (i % 12))
            
            raw_dur = float(features[i, 15].item()) * 4.0 if features.size(1) > 15 else 0.5
            dur_sec = max(0.15, raw_dur * 0.5) if raw_dur > 0 else 0.25
            
            score_notes.append({
                "pitch": pitch,
                "onset_sec": curr_onset,
                "duration_sec": dur_sec
            })
            curr_onset += 0.35

    # Clip predictions to match score_notes length (prevents endless audio rendering)
    num_notes = len(score_notes)
    predictions_clipped = {k: v[:num_notes] for k, v in predictions.items()}

    # A. Render Tenuto AI Expressive Performance MIDI
    render_expressive_midi(score_notes, predictions_clipped, output_midi_path=output_midi)

    # B. Render Mechanical Original Score MIDI (Un-expressed baseline)
    original_predictions = {
        "delta_t": torch.zeros(num_notes),
        "velocity": torch.full((num_notes,), 64.0),
        "articulation": torch.ones(num_notes),
        "pedal": torch.zeros(num_notes),
        "tempo_scale": torch.ones(num_notes)
    }
    render_expressive_midi(score_notes, original_predictions, output_midi_path="output_original.mid")

    # C. Locate or Copy Human Reference Performance MIDI (if available)
    try:
        import shutil
        human_midi_dst = "output_human.mid"
        if target_score and target_score.endswith(('.mid', '.midi')) and os.path.exists(target_score):
            shutil.copyfile(target_score, human_midi_dst)
            print(f"[Tenuto Inference] Found Human Reference MIDI: '{target_score}' -> '{human_midi_dst}'")
        else:
            score_dir = os.path.dirname(target_score) if target_score else ""
            if score_dir and os.path.exists(score_dir):
                for f in os.listdir(score_dir):
                    if f.endswith(('.mid', '.midi')) and not f.startswith('midi_score') and not f.startswith('output_'):
                        src_path = os.path.join(score_dir, f)
                        shutil.copyfile(src_path, human_midi_dst)
                        print(f"[Tenuto Inference] Found Human Reference MIDI: '{src_path}' -> '{human_midi_dst}'")
                        break
    except Exception as e:
        print(f"[Tenuto Inference] Human reference MIDI notice: {e}")

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
