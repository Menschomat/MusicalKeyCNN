import librosa
import numpy as np


def detect_bpm(audio_path, sample_rate=22050):
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    if len(beat_times) > 1:
        bpm = 60.0 / np.median(np.diff(beat_times))
    else:
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)
        bpm = float(np.atleast_1d(tempo)[0])
    return round(float(bpm), 1)
