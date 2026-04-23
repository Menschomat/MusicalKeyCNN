import librosa
import numpy as np
import torch


def load_audio(path, sample_rate=44100):
    """Load an audio file as a mono waveform at the given sample rate."""
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sr


def preprocess_from_waveform(waveform, sample_rate=44100, n_bins=105, hop_length=8820):
    """
    Extracts a log-magnitude CQT spectrogram from a pre-loaded mono waveform.

    Args:
        waveform (np.ndarray): Mono audio signal.
        sample_rate (int): Sample rate of the waveform.
        n_bins (int): Number of CQT bins.
        hop_length (int): Hop length for CQT.

    Returns:
        torch.Tensor: Shape (1, freq_bins, time_frames), ready for model input.
    """
    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=65)
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    return spec_tensor
