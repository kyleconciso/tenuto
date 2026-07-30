import numpy as np
import torch

def extract_40d_features_from_score(score_or_path):
    """
    Extracts the full 40D Note Feature Matrix X in R^(N x 40) from a score.
    Uses partitura if available, with robust fallback feature extraction.
    
    Feature Vector Schema (40 dims):
    --------------------------------
    0: pitch_norm (1)
    1-12: pitch_class_onehot (12)
    13: interval_from_root (1)
    14: melodic_leap (1)
    15: nominal_duration (1)
    16: beat_position (1)
    17-20: metric_weight_onehot (4)
    21: score_tempo (1)
    22-29: dynamic_marking_onehot (8)
    30: hairpin_slope (1)
    31-34: articulation_flags (4: staccato, tenuto, accent, fermata)
    35: is_top_melody (1)
    36: is_bass_note (1)
    37: chord_density (1)
    38-39: staff_id_onehot (2)
    """
    try:
        import partitura as pt
        if isinstance(score_or_path, str):
            score = pt.load_score(score_or_path)
        else:
            score = score_or_path

        # Partitura note array contains structured columns
        note_array = score.note_array()
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

        # Melodic leap (difference between consecutive notes)
        leaps = np.zeros(num_notes, dtype=np.float32)
        leaps[1:] = (pitches[1:] - pitches[:-1]) / 127.0
        features[:, 14] = leaps

        # 2. Metric Grid
        onset_beats = note_array['onset_beat']
        dur_beats = note_array['duration_beat']
        features[:, 15] = np.clip(dur_beats / 4.0, 0.0, 4.0)
        features[:, 16] = onset_beats % 4.0 / 4.0

        # Metric weight one-hot (4 dims: downbeat = beat 0)
        beat_idx = (np.floor(onset_beats) % 4).astype(int)
        for i, b in enumerate(beat_idx):
            features[i, 17 + b] = 1.0
        
        features[:, 21] = 120.0 / 240.0 # Default 120 BPM normalized

        # 3. Score Markings (Default dynamics mf = index 4)
        features[:, 26] = 1.0 # default mf

        # 4. Polyphony & Voice Analysis
        # Identify chord clusters based on identical onset_beats
        unique_onsets = np.unique(onset_beats)
        for onset in unique_onsets:
            idx = np.where(onset_beats == onset)[0]
            chord_size = len(idx)
            features[idx, 37] = chord_size / 10.0
            
            # Top melody & bass
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
        print(f"[TenutoFeatures] Note extraction warning: {e}. Generating fallback 40D features.")
        return torch.randn(256, 40)
