import librosa
import numpy as np
import torch

from constants import BINS_PER_OCTAVE, FMIN, HOP_LENGTH, N_BINS, SAMPLE_RATE, WAVEFORM_NUM_POINTS


def load_audio(path, sample_rate=SAMPLE_RATE):
    """Load an audio file as a mono waveform at the given sample rate."""
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sr


def compute_waveform(waveform, num_points=WAVEFORM_NUM_POINTS):
    """
    Downsample a waveform to num_points RMS values for visualization.
    Returns a list of floats normalized to [0, 1].
    """
    chunk_size = len(waveform) // num_points
    trimmed = waveform[: chunk_size * num_points]
    chunks = trimmed.reshape(num_points, chunk_size)
    rms = np.sqrt(np.mean(chunks ** 2, axis=1))
    max_val = rms.max() or 1.0
    return (rms / max_val).tolist()


def preprocess_from_waveform(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH):
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
    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN)
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    return spec_tensor
