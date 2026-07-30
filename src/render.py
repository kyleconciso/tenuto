import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np

def render_expressive_midi(score_notes, predictions, output_midi_path: str = "expressive_output.mid", base_bpm: float = 120.0):
    r"""
    Renders score notes + predicted expressive performance parameters into an expressive MIDI file.
    
    Includes:
      - Dynamic Tempo Track (S(b) rubato curve mapped to MIDI tempo events)
      - Micro-timed note onset and offset events (\Delta t_i and articulation d_i)
      - Dynamic velocity weighting v_i
      - Sustain Pedal CC64 control curves
    """
    mid = MidiFile(ticks_per_beat=480)
    
    # Track 0: Tempo track & Meta events
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('track_name', name='Tempo Track', time=0))

    # Track 1: Performance Notes & CC64 Pedal
    music_track = MidiTrack()
    mid.tracks.append(music_track)
    music_track.append(MetaMessage('track_name', name='Tenuto Expressive Performance', time=0))

    # Extract predictions
    delta_t = predictions['delta_t'].cpu().numpy() if hasattr(predictions['delta_t'], 'cpu') else predictions['delta_t']
    velocity = predictions['velocity'].cpu().numpy() if hasattr(predictions['velocity'], 'cpu') else predictions['velocity']
    articulation = predictions['articulation'].cpu().numpy() if hasattr(predictions['articulation'], 'cpu') else predictions['articulation']
    pedal = predictions['pedal'].cpu().numpy() if hasattr(predictions['pedal'], 'cpu') else predictions['pedal']
    
    tempo_scale = predictions.get('tempo_scale', np.ones(len(delta_t)))
    if hasattr(tempo_scale, 'cpu'):
        tempo_scale = tempo_scale.cpu().numpy()

    # Write initial tempo (microseconds per beat)
    initial_tempo = mido.bpm2tempo(base_bpm)
    tempo_track.append(MetaMessage('set_tempo', tempo=initial_tempo, time=0))

    events = []
    ticks_per_sec = 480 * (base_bpm / 60.0)

    num_notes = len(delta_t)
    for i in range(num_notes):
        pitch = int(score_notes[i]['pitch']) if isinstance(score_notes, (list, np.ndarray)) and 'pitch' in score_notes[i] else int(60 + (i % 24))
        onset_sec = score_notes[i]['onset_sec'] if isinstance(score_notes, (list, np.ndarray)) and 'onset_sec' in score_notes[i] else (i * 0.25)
        dur_sec = score_notes[i]['duration_sec'] if isinstance(score_notes, (list, np.ndarray)) and 'duration_sec' in score_notes[i] else 0.25
        
        # Apply predicted micro-timing shift and rubato scale
        scaled_onset_sec = (onset_sec * tempo_scale[i]) + delta_t[i]
        scaled_dur_sec = dur_sec * articulation[i]
        vel = int(np.clip(velocity[i], 1, 127))

        onset_tick = int(np.maximum(0, scaled_onset_sec * ticks_per_sec))
        offset_tick = int(np.maximum(onset_tick + 10, (scaled_onset_sec + scaled_dur_sec) * ticks_per_sec))

        events.append(('note_on', onset_tick, pitch, vel))
        events.append(('note_off', offset_tick, pitch, 0))

        # Add CC64 Sustain Pedal events
        if pedal[i] > 32:
            events.append(('control_change', onset_tick, 64, int(np.clip(pedal[i], 0, 127))))

    # Sort all events by absolute tick timestamp
    events.sort(key=lambda x: x[1])

    # Convert absolute tick timestamps into delta ticks for MIDI track
    last_tick = 0
    for ev_type, abs_tick, arg1, arg2 in events:
        delta_ticks = max(0, abs_tick - last_tick)
        last_tick = abs_tick

        if ev_type == 'note_on':
            music_track.append(Message('note_on', note=arg1, velocity=arg2, time=delta_ticks))
        elif ev_type == 'note_off':
            music_track.append(Message('note_off', note=arg1, velocity=arg2, time=delta_ticks))
        elif ev_type == 'control_change':
            music_track.append(Message('control_change', control=arg1, value=arg2, time=delta_ticks))

    mid.save(output_midi_path)
    print(f"[TenutoRender] Rendered expressive MIDI file saved to '{output_midi_path}'")
    return output_midi_path
