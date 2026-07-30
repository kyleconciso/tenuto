import numpy as np
import torch

def compute_alignment_targets(score_note_array, performance_note_array=None):
    r"""
    Derives ground-truth target features Y = [\Delta t_i, v_i, d_i, pedal_i, S(b)]
    from aligned score & performance note arrays.
    
    Target Schema:
    --------------
    1. Micro-Timing Shift (\Delta t_i): Played Time - Beat Grid Time bounded to [-25ms, +25ms]
    2. Note Velocity (v_i): MIDI velocity [0, 127]
    3. Articulation Scale (d_i): Played Duration / Score Duration
    4. Sustain Pedal (pedal_i): Continuous CC64 pedal values [0, 127]
    5. Beat Tempo Scale S(b): Macro rubato scale factor [0.5, 1.5]
    """
    num_notes = len(score_note_array) if hasattr(score_note_array, '__len__') else 256
    
    if performance_note_array is not None:
        # Extract ground truth from actual performance alignment
        delta_t = torch.tensor(performance_note_array.get('onset_sec', np.zeros(num_notes)) - score_note_array.get('onset_sec', np.zeros(num_notes)), dtype=torch.float32)
        delta_t = torch.clamp(delta_t, min=-0.025, max=0.025)
        
        velocity = torch.tensor(performance_note_array.get('velocity', np.full(num_notes, 64)), dtype=torch.float32)
        
        score_dur = score_note_array.get('duration_sec', np.ones(num_notes))
        perf_dur = performance_note_array.get('duration_sec', np.ones(num_notes))
        articulation = torch.tensor(perf_dur / np.maximum(score_dur, 1e-4), dtype=torch.float32)
        articulation = torch.clamp(articulation, min=0.1, max=3.0)
        
        pedal = torch.tensor(performance_note_array.get('pedal', np.zeros(num_notes)), dtype=torch.float32)
        tempo_scale = torch.tensor(performance_note_array.get('tempo_scale', np.ones(num_notes)), dtype=torch.float32)

    else:
        # Fallback target generation for synthetic testing
        delta_t = torch.clamp(0.01 * torch.randn(num_notes), min=-0.025, max=0.025)
        velocity = torch.clamp(64.0 + 15.0 * torch.randn(num_notes), min=0.0, max=127.0)
        articulation = torch.clamp(1.0 + 0.2 * torch.randn(num_notes), min=0.1, max=3.0)
        pedal = torch.clamp(64.0 + 30.0 * torch.randn(num_notes), min=0.0, max=127.0)
        tempo_scale = torch.clamp(1.0 + 0.1 * torch.randn(num_notes), min=0.5, max=1.5)

    return {
        "delta_t": delta_t,
        "velocity": velocity,
        "articulation": articulation,
        "pedal": pedal,
        "tempo_scale": tempo_scale
    }
