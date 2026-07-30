import os
import subprocess

def render_midi_to_wav(midi_path: str, wav_path: str = None):
    """
    Renders a MIDI file to WAV audio using fluidsynth, timidity, or midi2audio.
    """
    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    # 1. Try FluidSynth command line directly
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
            if os.path.exists(wav_path):
                print(f"[TenutoAudio] FluidSynth synthesized '{midi_path}' -> '{wav_path}'")
                return wav_path
        except Exception:
            pass

    # 2. Try TiMidity command line fallback
    try:
        cmd = ["timidity", midi_path, "-Ow", "-o", wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(wav_path):
            print(f"[TenutoAudio] TiMidity synthesized '{midi_path}' -> '{wav_path}'")
            return wav_path
    except Exception:
        pass

    # 3. Try midi2audio python package
    try:
        from midi2audio import FluidSynth
        fs = FluidSynth()
        fs.midi_to_audio(midi_path, wav_path)
        if os.path.exists(wav_path):
            print(f"[TenutoAudio] midi2audio synthesized '{midi_path}' -> '{wav_path}'")
            return wav_path
    except Exception:
        pass

    print(f"[TenutoAudio] Synthesis notice: FluidSynth/TiMidity binary not found. Install via `apt-get install -y fluidsynth fluid-soundfont-gm`.")
    return None

def play_audio_in_colab(wav_or_midi_path: str, title: str = "Audio Player"):
    """
    Embeds an interactive audio player in Jupyter / Google Colab notebook.
    """
    try:
        from IPython.display import display, Audio
        print(f"--- Playing: {title} ---")
        
        # If input is XML or score, convert to MIDI first or search for generated MIDI
        if wav_or_midi_path.endswith(('.xml', '.mxl', '.musicxml')):
            print(f"[TenutoAudio] Score file '{wav_or_midi_path}' passed. Displaying score player info.")
            return

        wav_path = render_midi_to_wav(wav_or_midi_path)
        if wav_path and os.path.exists(wav_path):
            display(Audio(wav_path))
        else:
            print(f"[TenutoAudio] Playing MIDI path directly: '{wav_or_midi_path}'")
    except ImportError:
        print(f"[TenutoAudio] IPython display not available.")
