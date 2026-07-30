import os
import subprocess
import numpy as np

def render_midi_to_wav(midi_path: str, wav_path: str = None):
    """
    Renders a MIDI file to WAV audio using FluidSynth, TiMidity, or fallback synthesis.
    """
    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    # 1. Try FluidSynth command line
    soundfont_paths = [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/default.sf2",
        "/usr/share/sounds/sf2/TimGM6mb.sf2"
    ]
    sf2 = next((sf for sf in soundfont_paths if os.path.exists(sf)), None)

    if sf2:
        try:
            cmd = ["fluidsynth", "-ni", sf2, midi_path, "-F", wav_path, "-r", "44100"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                print(f"[TenutoAudio] FluidSynth synthesized '{midi_path}' -> '{wav_path}'")
                return wav_path
        except Exception:
            pass

    # 2. Try TiMidity command line fallback
    try:
        cmd = ["timidity", midi_path, "-Ow", "-o", wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
            print(f"[TenutoAudio] TiMidity synthesized '{midi_path}' -> '{wav_path}'")
            return wav_path
    except Exception:
        pass

    # 3. Built-in Sine Synth Fallback (Guarantees playable WAV Audio Player in Colab)
    try:
        import mido
        from scipy.io import wavfile
        
        mid = mido.MidiFile(midi_path)
        sample_rate = 22050
        duration = max(5.0, mid.length)
        total_samples = int(sample_rate * duration)
        audio = np.zeros(total_samples, dtype=np.float32)

        ticks_per_sec = 480 * 2 # Default tempo
        current_time = 0.0

        for track in mid.tracks:
            t_sec = 0.0
            for msg in track:
                t_sec += msg.time / 480.0
                if msg.type == 'note_on' and msg.velocity > 0:
                    freq = 440.0 * (2.0 ** ((msg.note - 69) / 12.0))
                    note_dur = 0.4
                    start_idx = int(t_sec * sample_rate)
                    end_idx = min(total_samples, int((t_sec + note_dur) * sample_rate))
                    if start_idx < total_samples and end_idx > start_idx:
                        t = np.linspace(0, note_dur, end_idx - start_idx, False)
                        # Simple harmonic piano tone
                        wave = (0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)) * np.exp(-3 * t)
                        audio[start_idx:end_idx] += wave * (msg.velocity / 127.0)

        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio_int16 = (audio * 32767).astype(np.int16)

        wavfile.write(wav_path, sample_rate, audio_int16)
        print(f"[TenutoAudio] Built-in audio synth rendered '{midi_path}' -> '{wav_path}'")
        return wav_path

    except Exception as e:
        print(f"[TenutoAudio] Synthesis fallback notice: {e}")

    return None

def play_audio_in_colab(wav_or_midi_path: str, title: str = "Audio Player"):
    """
    Embeds an interactive HTML5 / IPython Audio player directly in Colab.
    """
    try:
        from IPython.display import display, Audio, HTML
        print(f"--- Playing: {title} ---")
        
        wav_path = render_midi_to_wav(wav_or_midi_path)
        if wav_path and os.path.exists(wav_path):
            display(Audio(wav_path, rate=22050))
            display(HTML(f"<b>🎵 Play Audio ({title}):</b> <code>{os.path.basename(wav_path)}</code>"))
        else:
            print(f"[TenutoAudio] Could not generate WAV for: '{wav_or_midi_path}'")
    except Exception as e:
        print(f"[TenutoAudio] Audio player error: {e}")
