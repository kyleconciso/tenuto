import numpy as np
import torch

def extract_40d_features_from_score(score_or_path):
    """
    Extracts the full 40D Note Feature Matrix X in R^(N x 40) from a score.
    Uses partitura if available, with multi-part division handling.
    """
    try:
        import partitura as pt
        if isinstance(score_or_path, str):
            score = pt.load_score(score_or_path)
        else:
            score = score_or_path

        # Safely extract note array (handle multi-division parts if score.note_array() fails)
        try:
            note_array = score.note_array()
        except Exception:
            note_arrays = []
            for part in score.parts:
                try:
                    na = part.note_array()
                    if na is not None and len(na) > 0:
                        note_arrays.append(na)
                except Exception:
                    pass
            if note_arrays:
                note_array = np.concatenate(note_arrays)
            else:
                note_array = np.array([])

        num_notes = len(note_array)
        if num_notes == 0:
            return torch.zeros(0, 40)

        features = np.zeros((num_notes, 40), dtype=np.float32)

        # 1. Pitch & Harmony
        pitches = note_array['pitch']
        features[:, 0] = pitches / 127.0
        
        # Pitch class one-hot (12 dims)
        pitch_classes = pitches % 12
        for i, pc in enumerate(pitch_classes):
            features[i, 1 + pc] = 1.0

        # Melodic leap
        leaps = np.zeros(num_notes, dtype=np.float32)
        leaps[1:] = (pitches[1:] - pitches[:-1]) / 127.0
        features[:, 14] = leaps

        # 2. Metric Grid
        onset_beats = note_array['onset_beat'] if 'onset_beat' in note_array.dtype.names else np.zeros(num_notes)
        dur_beats = note_array['duration_beat'] if 'duration_beat' in note_array.dtype.names else np.ones(num_notes)
        features[:, 15] = np.clip(dur_beats / 4.0, 0.0, 4.0)
        features[:, 16] = onset_beats % 4.0 / 4.0

        # Metric weight one-hot
        beat_idx = (np.floor(onset_beats) % 4).astype(int)
        for i, b in enumerate(beat_idx):
            features[i, 17 + b] = 1.0
        
        features[:, 21] = 120.0 / 240.0 # Default 120 BPM normalized

        # 3. Score Markings (Default mf = index 26)
        features[:, 26] = 1.0

        # 4. Polyphony & Voice Analysis
        unique_onsets = np.unique(onset_beats)
        for onset in unique_onsets:
            idx = np.where(onset_beats == onset)[0]
            chord_size = len(idx)
            features[idx, 37] = chord_size / 10.0
            
            top_idx = idx[np.argmax(pitches[idx])]
            bass_idx = idx[np.argmin(pitches[idx])]
            features[top_idx, 35] = 1.0
            features[bass_idx, 36] = 1.0

        # Staff ID
        if 'staff' in note_array.dtype.names:
            staffs = note_array['staff']
            for i, st in enumerate(staffs):
                st_idx = 0 if st <= 1 else 1
                features[i, 38 + st_idx] = 1.0
        else:
            features[:, 38] = 1.0

        return torch.tensor(features, dtype=torch.float32)

    except Exception as e:
        print(f"[Features] Failed to extract features: {e}")
        return None
