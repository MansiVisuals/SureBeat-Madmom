import os
import sys
import subprocess
from madmom.audio.signal import Signal
from madmom.features.onsets import OnsetPeakPickingProcessor, CNNOnsetProcessor
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

def convert_to_wav(input_path):
    """Convert input audio file to .wav using FFmpeg and return the new file path."""
    output_path = '/tmp/converted_audio.wav'
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le", "-ar", "44100", output_path
    ]
    print("Converting audio to wav format for processing...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with error:\n{result.stderr}")
    print("Conversion successful. Temporary wav file created at:", output_path)
    return output_path

def detect_beats_and_tempo(audio_path):
    # Convert audio to WAV format for processing
    wav_path = convert_to_wav(audio_path)

    # Load the converted WAV file
    print("Loading WAV file for processing:", wav_path)
    signal = Signal(wav_path, sample_rate=44100)

    # Beat Detection (formerly onset detection)
    print("Detecting beats...")
    beat_processor = CNNOnsetProcessor()
    peak_picking = OnsetPeakPickingProcessor(fps=100, threshold=0.5, pre_avg=0.15, post_avg=0.15, pre_max=0.05, post_max=0.05)
    beat_activations = beat_processor(signal)
    beats = peak_picking(beat_activations)

    # Tempo Detection (formerly beat detection)
    print("Detecting tempo...")
    tempo_processor = RNNBeatProcessor()(wav_path)
    dbn_processor = DBNBeatTrackingProcessor(min_bpm=30, max_bpm=200, fps=100)
    tempo = dbn_processor(tempo_processor)

    # Clean up temporary WAV file
    print("Temporary wav file removed.")
    os.remove(wav_path)

    # Output results
    print("Detected Beats:")
    for beat_time in beats:
        print(f"BEAT: {beat_time}")

    print("Detected Tempos:")
    for tempo_time in tempo:
        print(f"TEMPO: {tempo_time}")

if __name__ == "__main__":
    # Debug information for environment
    print("Python Environment Information:")
    print("Python version:", sys.version)
    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")
    print("Current Working Directory:", os.getcwd())

    if len(sys.argv) < 2:
        print("Usage: python beat_detection.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    try:
        detect_beats_and_tempo(audio_path)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
