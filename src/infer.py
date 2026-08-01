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

def extract_piece_metadata(target_score: str, pt_data: dict = None):
    """Extracts human-readable composer, piece title, and metadata from file path or dataset dict."""
    composer = "Unknown Composer"
    piece = "Unknown Piece"
    source = os.path.basename(target_score)

    ref_path = target_score
    if pt_data and isinstance(pt_data, dict):
        ref_path = pt_data.get("score_path") or pt_data.get("perf_path") or target_score
        source = pt_data.get("source") or os.path.basename(target_score)

    parts = ref_path.replace("\\", "/").split("/")
    if "asap" in parts:
        try:
            idx = parts.index("asap")
            if idx + 2 < len(parts):
                composer = parts[idx + 1]
                piece_parts = parts[idx + 2:-1]
                piece = " - ".join(piece_parts) if piece_parts else parts[idx + 2]
        except Exception:
            pass
    elif "pianocore" in parts or "pianocore" in str(source).lower():
        composer = "PianoCoRe Dataset Pair"
        piece = os.path.splitext(os.path.basename(ref_path))[0]
    else:
        piece = os.path.splitext(os.path.basename(target_score))[0]

    return {
        "composer": composer.replace("_", " "),
        "piece": piece.replace("_", " "),
        "source": source
    }

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
        
        # Priority 1: Search for real score / MIDI files (.musicxml, .xml, .mxl, .mid, .midi)
        for sroot in search_roots:
            if os.path.exists(sroot):
                for root, _, files in os.walk(sroot):
                    for f in files:
                        if f.lower().endswith(('.xml', '.mxl', '.musicxml', '.mid', '.midi')):
                            target_score = os.path.join(root, f)
                            print(f"[Tenuto Inference] Score '{score_path}' not found. Using dataset score file '{target_score}'.")
                            found = True
                            break
                    if found:
                        break
            if found:
                break
                
        # Priority 2: Fallback to preprocessed .pt files if no raw score/MIDI exists
        if not found:
            for sroot in search_roots:
                if os.path.exists(sroot):
                    for root, _, files in os.walk(sroot):
                        for f in files:
                            if f.lower().endswith('.pt'):
                                target_score = os.path.join(root, f)
                                print(f"[Tenuto Inference] Score '{score_path}' not found. Using preprocessed tensor file '{target_score}'.")
                                found = True
                                break
                        if found:
                            break
                if found:
                    break

        if not os.path.exists(target_score):
            raise FileNotFoundError(f"[Tenuto Inference] Error: Score file '{score_path}' not found, and no dataset files exist in ./data. Please provide a valid score path.")

    # 1. Extract 40D Features
    features = None
    pt_targets = None
    raw_pt_data = None
    if os.path.exists(target_score):
        if target_score.endswith('.pt'):
            try:
                pt_data = torch.load(target_score, map_location="cpu")
                raw_pt_data = pt_data
                if isinstance(pt_data, dict):
                    features = pt_data.get("x")
                    pt_targets = pt_data.get("targets")
                else:
                    features = pt_data
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
                                raw_pt_data = pt_data
                                if isinstance(pt_data, dict):
                                    pt_x = pt_data.get("x")
                                    pt_targets = pt_data.get("targets")
                                else:
                                    pt_x = pt_data
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
        raise ValueError(f"[Tenuto Inference] Error: Could not extract features from '{target_score}'. File is missing, corrupt, or invalid score format.")

    # Extract & Print Piece Metadata
    meta = extract_piece_metadata(target_score, raw_pt_data)
    print("=" * 70)
    print(f"🎼 TENUTO INFERENCE PIECE METADATA")
    print(f"  • Composer      : {meta['composer']}")
    print(f"  • Piece / Title : {meta['piece']}")
    print(f"  • Source File   : {target_score}")
    print(f"  • Total Notes   : {features.size(0):,} notes")
    print("=" * 70)
    
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
        print("[Tenuto Inference] Reconstructing polyphonic note sequence from feature matrix...")
        num_notes = min(features.size(0), 256)
        score_notes = []
        curr_onset = 0.0
        
        for i in range(num_notes):
            raw_pitch = float(features[i, 0].item()) * 127.0 if features.size(1) > 0 else 60.0
            pitch = int(np.clip(round(raw_pitch), 21, 108)) if raw_pitch > 10 else int(60 + (i % 12))
            
            raw_dur = float(features[i, 15].item()) * 4.0 if features.size(1) > 15 else 0.5
            dur_sec = max(0.2, raw_dur * 0.5)
            
            # Polyphony / chord size detection from feature index 37
            chord_size = int(round(float(features[i, 37].item()) * 10.0)) if features.size(1) > 37 else 1
            chord_size = max(1, min(chord_size, 4))
            
            score_notes.append({
                "pitch": pitch,
                "onset_sec": curr_onset,
                "duration_sec": dur_sec
            })
            
            # Group notes in chords together, advance onset at lively musical tempo (0.14s per step)
            if (i + 1) % chord_size == 0:
                curr_onset += 0.14

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

    # C. Locate or Render Human Reference Performance MIDI (if available)
    try:
        human_midi_dst = "output_human.mid"
        if pt_targets:
            pt_targets_clipped = {k: v[:num_notes] for k, v in pt_targets.items()}
            render_expressive_midi(score_notes, pt_targets_clipped, output_midi_path=human_midi_dst)
            print(f"[Tenuto Inference] Rendered Ground-Truth Human Reference from dataset -> '{human_midi_dst}'")
        elif target_score and target_score.endswith(('.mid', '.midi')) and os.path.exists(target_score):
            import shutil
            shutil.copyfile(target_score, human_midi_dst)
            print(f"[Tenuto Inference] Found Human Reference MIDI: '{target_score}' -> '{human_midi_dst}'")
        else:
            score_dir = os.path.dirname(target_score) if target_score else ""
            if score_dir and os.path.exists(score_dir):
                import shutil
                for f in os.listdir(score_dir):
                    if f.endswith(('.mid', '.midi')) and not f.startswith('midi_score') and not f.startswith('output_'):
                        src_path = os.path.join(score_dir, f)
                        shutil.copyfile(src_path, human_midi_dst)
                        print(f"[Tenuto Inference] Found Human Reference MIDI: '{src_path}' -> '{human_midi_dst}'")
                        break
    except Exception as e:
        print(f"[Tenuto Inference] Human reference MIDI notice: {e}")

    predictions["_metadata"] = meta
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
