import librosa
import numpy as np


def detect_bpm(audio_path, sample_rate=22050):
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    return round(float(np.atleast_1d(tempo)[0]), 1)
