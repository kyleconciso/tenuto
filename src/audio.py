import os

def render_midi_to_wav(midi_path: str, wav_path: str = None):
    """
    Renders a MIDI file to WAV audio using fluidsynth / midi2audio if available.
    """
    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    try:
        from midi2audio import FluidSynth
        fs = FluidSynth()
        fs.midi_to_audio(midi_path, wav_path)
        print(f"[TenutoAudio] Synthesized '{midi_path}' -> '{wav_path}'")
        return wav_path
    except Exception as e:
        print(f"[TenutoAudio] FluidSynth audio synthesis notice: {e}")
        return None

def play_audio_in_colab(wav_or_midi_path: str, title: str = "Audio Player"):
    """
    Embeds an interactive audio player in Jupyter / Google Colab notebook.
    """
    try:
        from IPython.display import display, Audio, HTML
        print(f"--- Playing: {title} ---")
        wav_path = render_midi_to_wav(wav_or_midi_path)
        if wav_path and os.path.exists(wav_path):
            display(Audio(wav_path))
        else:
            print(f"[TenutoAudio] Displaying MIDI file path: '{wav_or_midi_path}'")
    except ImportError:
        print(f"[TenutoAudio] IPython display not available.")
